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
        """.hef가 내보내는 출력 텐서의 이름/shape/포맷을 사람이 읽을 수 있게 정리.

        decode_keypoints()를 구현하려면 먼저 이 정보가 필요하다. 출력이 어떤
        형태인지에 따라 후처리가 완전히 달라지기 때문이다:
          - NMS까지 포함해 컴파일된 .hef  -> 검출 결과가 바로 나와 디코딩이 거의 불필요
          - raw 텐서로 컴파일된 .hef      -> DFL 디스트리뷰션 디코딩 + NMS를 직접 해야 함
        Pi5에서 아래로 확인한다:
            python -c "from pathlib import Path; from rpi.pose_hailo import HailoPoseEstimator; \
                       print(HailoPoseEstimator(Path('rpi/yolov8_pose.hef')).describe_outputs())"
        """
        lines = [f"input : {self.input_info.name}  shape={self.input_info.shape}"]
        for info in self.output_infos:
            fmt = getattr(getattr(info, "format", None), "type", None)
            lines.append(f"output: {info.name}  shape={info.shape}"
                         + (f"  format={fmt}" if fmt is not None else ""))
        lines.append(f"출력 텐서 개수: {len(self.output_infos)}")
        return "\n".join(lines)

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
