"""
YOLOv8-pose(COCO-17 keypoint) 출력으로부터 자세 상태를 추정하는 규칙 기반 휴리스틱.

AI-Hub 데이터에 실제 전신 자세 라벨이 없어 학습된 모델이 아니라 기하학적 규칙을 쓴다.
임계값은 실제 카메라 설치 각도/거리에 맞춰 캘리브레이션이 필요하다.
"""
from dataclasses import dataclass

# COCO-17 keypoint 인덱스 (YOLOv8-pose 기본 출력 순서)
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12


@dataclass
class PostureResult:
    label: str          # "바른 자세" | "구부정함" | "엎드림" | "기울어짐" | "판단불가"
    shoulder_tilt: float
    head_drop_ratio: float


def classify_posture(keypoints: list[tuple[float, float, float]],
                      conf_th: float = 0.4) -> PostureResult:
    """keypoints: 길이 17의 (x, y, confidence) 리스트 (YOLOv8-pose 출력 그대로)."""
    def pt(idx):
        x, y, c = keypoints[idx]
        return (x, y) if c >= conf_th else None

    nose = pt(NOSE)
    l_sh, r_sh = pt(L_SHOULDER), pt(R_SHOULDER)

    if not (nose and l_sh and r_sh):
        return PostureResult("판단불가", 0.0, 0.0)

    shoulder_mid = ((l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2)
    shoulder_width = max(abs(l_sh[0] - r_sh[0]), 1e-3)

    # 어깨 좌우 기울기 (좌우 y 차이 / 어깨너비)
    shoulder_tilt = abs(l_sh[1] - r_sh[1]) / shoulder_width

    # 머리가 어깨선 아래로 내려온 정도 (엎드림/구부정 감지)
    head_drop_ratio = (nose[1] - shoulder_mid[1]) / shoulder_width

    if head_drop_ratio > 0.3:
        label = "엎드림"
    elif shoulder_tilt > 0.25:
        label = "기울어짐"
    elif head_drop_ratio > 0.05:
        label = "구부정함"
    else:
        label = "바른 자세"

    return PostureResult(label, shoulder_tilt, head_drop_ratio)
