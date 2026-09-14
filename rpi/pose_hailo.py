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

    def decode_keypoints(self, raw_outputs: dict) -> list[list[tuple[float, float, float]]]:
        """*** 미검증 ***: hailo-rpi5-examples의 실제 후처리 로직으로 교체 필요."""
        raise NotImplementedError(
            "YOLOv8-pose 출력 텐서 디코딩은 hailo-rpi5-examples의 공식 후처리 코드를 "
            "이식해서 구현하세요. raw_outputs의 키/shape를 print로 확인 후 "
            "해당 저장소의 postprocess 함수와 대조하는 것을 권장합니다."
        )
