"""
Hailo-8L(라즈베리파이5 AI Kit)에서 사전 컴파일된 YOLOv8-pose(.hef)를 돌려
COCO-17 키포인트를 뽑아내는 래퍼.

*** 주의: 이 파일은 실제 Hailo-8L 하드웨어에서 검증되지 않았습니다. ***
HailoRT Python API의 전체적인 사용 패턴(VDevice/HEF/ConfigureParams/InferVStreams)은
Hailo 공식 문서 기준으로 작성했지만, YOLOv8-pose의 raw 출력 텐서를 실제 키포인트
좌표로 디코딩하는 후처리 로직은 모델 버전/컴파일 옵션에 따라 달라질 수 있어
자신 있게 구현하지 못했습니다.

권장 절차:
1. https://github.com/hailo-ai/hailo-rpi5-examples 공식 예제 저장소를 Pi5에 클론
2. 그 저장소의 pose estimation 예제(GStreamer 파이프라인 + 후처리 .so/.py)를 먼저
   그대로 돌려서 정상 동작을 확인
3. 그 예제의 후처리 함수를 아래 decode_keypoints()에 그대로 옮겨오거나,
   이 클래스 대신 그 예제의 파이프라인을 직접 호출하도록 rpi/pi_main.py를 수정
"""
from pathlib import Path

import numpy as np


class HailoPoseEstimator:
    def __init__(self, hef_path: Path):
        from hailo_platform import (
            HEF, VDevice, HailoStreamInterface, ConfigureParams,
            InputVStreamParams, OutputVStreamParams, FormatType,
        )

        self.hef = HEF(str(hef_path))
        self.target = VDevice()
        configure_params = ConfigureParams.create_from_hef(
            self.hef, interface=HailoStreamInterface.PCIe
        )
        self.network_group = self.target.configure(self.hef, configure_params)[0]
        self.network_group_params = self.network_group.create_params()

        self.input_info = self.hef.get_input_vstream_infos()[0]
        self.output_infos = self.hef.get_output_vstream_infos()
        self.input_vstreams_params = InputVStreamParams.make(
            self.network_group, format_type=FormatType.FLOAT32
        )
        self.output_vstreams_params = OutputVStreamParams.make(
            self.network_group, format_type=FormatType.FLOAT32
        )
        self._input_shape = self.input_info.shape  # (H, W, C) 보통

    def infer(self, frame_rgb: np.ndarray) -> list[list[tuple[float, float, float]]]:
        """frame_rgb: 모델 입력 해상도로 이미 리사이즈된 RGB 프레임.
        반환: 검출된 사람별 [(x, y, conf) x 17] 리스트. (후처리 미검증 — 아래 decode_keypoints 참고)"""
        from hailo_platform import InferVStreams

        with self.network_group.activate(self.network_group_params):
            with InferVStreams(self.network_group, self.input_vstreams_params,
                                self.output_vstreams_params) as pipeline:
                input_data = {self.input_info.name: np.expand_dims(frame_rgb, 0).astype(np.float32)}
                raw_outputs = pipeline.infer(input_data)

        return self.decode_keypoints(raw_outputs)

    def describe_outputs(self) -> str:
        """.hef가 내보내는 출력 텐서의 이름/shape/포맷을 정리하고, 어떤 후처리가
        필요한 레이아웃인지까지 판정한다.

        decode_keypoints()를 구현하려면 먼저 이 정보가 필요하다. 출력이 어떤
        형태인지에 따라 후처리가 완전히 달라지기 때문이다:
          - NMS까지 포함해 컴파일된 .hef  -> 검출 결과가 바로 나와 디코딩이 거의 불필요
          - raw 텐서로 컴파일된 .hef      -> DFL 디스트리뷰션 디코딩 + NMS를 직접 해야 함
        Pi5에서 아래로 확인한다:
            python -c "from pathlib import Path; from rpi.pose_hailo import HailoPoseEstimator; \
                       print(HailoPoseEstimator(Path('rpi/yolov8_pose.hef')).describe_outputs())"

        판정 결과를 그대로 옮겨 붙여 이슈에 남기면, 하드웨어 없는 쪽에서도
        어느 디코더를 써야 하는지 판단할 수 있다.
        """
        lines = [f"input : {self.input_info.name}  shape={self.input_info.shape}"]
        shapes = []
        for info in self.output_infos:
            fmt = getattr(getattr(info, "format", None), "type", None)
            shape = tuple(info.shape)
            shapes.append(shape)
            lines.append(f"output: {info.name}  shape={shape}"
                         + (f"  format={fmt}" if fmt is not None else ""))
        lines.append(f"출력 텐서 개수: {len(self.output_infos)}")
        lines.append("")
        lines.append(self._classify_layout(shapes))
        return "\n".join(lines)

    def _classify_layout(self, shapes: list[tuple]) -> str:
        """출력 shape로 후처리 방식을 판정한다.

        YOLOv8-pose(COCO-17, 사람 1클래스)의 채널 수는 정해져 있다:
          - 박스 분포(DFL): reg_max=16, 변 4개 -> 64채널
          - 클래스 점수   : 1채널(person)
          - 키포인트      : 17 x (x, y, conf) -> 51채널
        raw로 컴파일되면 스트라이드 8/16/32에 대해 위 채널들이 따로 나온다.
        NMS 포함으로 컴파일되면 검출 개수 축을 가진 텐서 하나로 나온다.
        """
        n = len(shapes)
        flat = [s for s in shapes]
        chans = {s[-1] for s in flat if len(s) >= 1}

        hints = []
        # raw 텐서 판정: 64/1/51 채널이 보이면 DFL 디코딩이 필요하다.
        raw_markers = {64, 51}
        if raw_markers & chans:
            hints.append(
                "판정: raw 텐서 (NMS 미포함)로 보인다. 64채널=DFL 박스 분포, "
                "51채널=키포인트 17x3, 1채널=person 점수.\n"
                "  -> DFL softmax·기댓값으로 박스를 복원하고, 스트라이드별 그리드 앵커를 "
                "더한 뒤 NMS를 직접 돌려야 한다.\n"
                "  -> hailo-rpi5-examples의 pose 후처리(.so 또는 파이썬 구현)를 "
                "그대로 이식하는 편이 안전하다."
            )
        # NMS 포함 판정: 마지막 축이 검출 속성 개수(보통 6 또는 56=4+1+51)인 경우.
        if chans & {6, 56, 57}:
            hints.append(
                "판정: NMS 포함으로 보인다(마지막 축이 검출 속성 개수). "
                "박스·점수·키포인트가 이미 디코딩된 상태일 가능성이 높다.\n"
                "  -> 좌표가 0~1 정규화인지 픽셀 단위인지만 확인하면 거의 그대로 쓸 수 있다."
            )
        if n >= 6 and not hints:
            hints.append(
                f"판정: 출력이 {n}개다. 스트라이드 3개 x (박스/점수/키포인트) = 9개 구조일 "
                "가능성이 있다. 각 텐서의 마지막 축 채널 수로 역할을 가려야 한다."
            )
        if not hints:
            hints.append(
                "판정: 알려진 패턴에 맞지 않는다. 채널 수가 64(DFL)·51(키포인트)·1(점수)와 "
                "다르면 모델 변형이거나 양자화 출력이 합쳐진 경우다. "
                "hailo-rpi5-examples 예제를 먼저 돌려 정상 동작을 확인할 것."
            )

        hints.append(f"\n관측된 마지막 축 채널 수: {sorted(chans)}")
        return "\n".join(hints)

    def decode_keypoints(self, raw_outputs: dict) -> list[list[tuple[float, float, float]]]:
        """*** 미구현 ***: hailo-rpi5-examples의 실제 후처리 로직을 이식해야 한다.

        추측으로 구현하지 않는 이유: 출력 텐서 레이아웃이 모델 버전과 컴파일
        옵션(NMS 포함 여부, 입력 해상도, 양자화 스케일)에 따라 달라진다.
        잘못 디코딩하면 예외 대신 '그럴듯하지만 틀린 좌표'가 나오고, 그 값이
        classify_posture()를 거쳐 백엔드 학습활동 기록까지 그대로 올라간다.
        조용히 틀린 데이터가 쌓이는 것보다 여기서 멈추는 편이 낫다.

        구현 절차:
        1. Pi5에서 describe_outputs()로 실제 텐서 이름/shape 확인
        2. hailo-ai/hailo-rpi5-examples 의 pose estimation 예제를 그대로 돌려
           정상 동작을 먼저 확인
        3. 그 예제의 후처리 함수를 여기로 옮기고, 좌표를 (x, y, conf) 17개
           튜플 리스트로 변환해 반환 (COCO-17 순서 유지)
        """
        raise NotImplementedError(
            "YOLOv8-pose 출력 디코딩 미구현. describe_outputs()로 텐서 형태를 확인한 뒤 "
            "hailo-rpi5-examples의 공식 후처리 코드를 이식하세요. "
            f"현재 출력 텐서: {[i.name for i in self.output_infos]}"
        )
