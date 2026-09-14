"""
라즈베리파이5 실행 진입점: 카메라 → (집중도 분류 + 자세 추정) → 상태 출력.

카메라/dlib/Hailo 등 Pi5 전용 의존성이 필요해 개발 PC에서는 실행할 수 없다.
Pi5에서:
    pip install dlib opencv-python joblib
    python rpi/pi_main.py
포즈 추정(HailoPoseEstimator)은 pose_hailo.py의 decode_keypoints()가 구현되기 전까지
자동으로 비활성화되고 집중도 분류만 동작한다.
"""
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from rpi.live_attention import AttentionEstimator
from rpi.posture_heuristic import classify_posture

POSE_HEF_PATH = Path(__file__).parent / "yolov8_pose.hef"
INFER_INTERVAL_SEC = 1.0  # AI-Hub 라벨의 10초 구간보다 촘촘하게, 1초마다 판단


def try_load_pose_estimator():
    if not POSE_HEF_PATH.exists():
        print(f"[Pose] .hef 없음({POSE_HEF_PATH}) — 자세 추정 비활성화")
        return None
    try:
        from rpi.pose_hailo import HailoPoseEstimator
        return HailoPoseEstimator(POSE_HEF_PATH)
    except Exception as e:
        print(f"[Pose] 초기화 실패({e}) — 자세 추정 비활성화")
        return None


def main():
    attention = AttentionEstimator()
    pose_estimator = try_load_pose_estimator()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("카메라를 열 수 없습니다 (index 0)")

    last_infer = 0.0
    print("실행 시작 (Ctrl+C로 종료)")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            now = time.time()
            if now - last_infer < INFER_INTERVAL_SEC:
                continue
            last_infer = now

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            result = attention.estimate(gray)

            status = {"time": time.strftime("%H:%M:%S")}
            if result:
                label, proba = result
                status["집중도"] = label
                status["확률"] = {k: round(v, 3) for k, v in proba.items()}
            else:
                status["집중도"] = "얼굴 미검출"

            if pose_estimator:
                try:
                    resized = cv2.resize(frame, pose_estimator._input_shape[:2][::-1])
                    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    people = pose_estimator.infer(rgb)
                    if people:
                        posture = classify_posture(people[0])
                        status["자세"] = posture.label
                except NotImplementedError:
                    pass
                except Exception as e:
                    print(f"[Pose] 추론 오류: {e}")

            print(status)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()


if __name__ == "__main__":
    main()
