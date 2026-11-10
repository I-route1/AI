"""
라즈베리파이5 실행 진입점: 카메라 → (집중도 분류 + 자세 추정) → 상태 출력.

카메라/dlib/Hailo 등 Pi5 전용 의존성이 필요해 개발 PC에서는 실행할 수 없다.
Pi5에서:
    pip install dlib opencv-python joblib
    python rpi/pi_main.py
Hailo 모델은 POSE_HEF_PATH 또는 공식 hailo-models 설치 경로에서 읽는다.
얼굴 측정을 제외하고 자세만 분석하려면 --pose-only를 사용한다(pose_main.py 인자 지원).

종료(Ctrl+C) 시 세션을 집계해 백엔드 POST /api/activities 로 올리려면:
    python rpi/pi_main.py --upload --student-id 14 --subject 수학 --understanding 4
이때 .env에 BACKEND_URL과 토큰(또는 계정)이 있어야 한다. pi_backend.py 참고.

두 점수 모두 Backend LearningActivity의 1~5점 별점 척도로 보낸다. Backend는 앱에서
학생이 매긴 별점과 같은 필드로 받고, 메타인지 분석은 이해도에 20을 곱해 100점으로
환산한다. 예전에는 집중도를 0~100으로, 이해도를 기본값 0으로 보내 평균이 왜곡됐다.
- 집중도(concentrationScore): '집중' 판정 비율 0~100%를 1~5점으로 환산(_to_stars)
- 이해도(understandingScore): 카메라로 측정할 수 없어 추정하지 않는다. --upload 때는
  --understanding 1~5를 반드시 받는다(0을 보내면 이해도 평균을 끌어내린다).
"""
import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from rpi.posture_session import PostureSession

INFER_INTERVAL_SEC = 1.0  # AI-Hub 라벨의 10초 구간보다 촘촘하게, 1초마다 판단
FOCUSED_LABEL = "집중"    # rpi/dataset.py CATEGORY_NAMES = ["집중", "집중하지않음", "졸음"]


def try_load_pose_estimator():
    from rpi.pose_hailo import default_hef_path
    path = default_hef_path()
    if not path.exists():
        print(f"[Pose] .hef 없음({path}) — 자세 추정 비활성화")
        return None
    try:
        from rpi.pose_hailo import HailoPoseEstimator
        return HailoPoseEstimator(path)
    except Exception as e:
        print(f"[Pose] 초기화 실패({e}) — 자세 추정 비활성화")
        return None


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true", help="종료 시 세션을 백엔드에 기록")
    ap.add_argument("--student-id", type=int, default=None,
                    help="업로드 대상 학생 ID (기본: .env의 STUDENT_ID)")
    ap.add_argument("--subject", default=None, help="업로드 시 과목명 (예: 수학)")
    ap.add_argument("--understanding", type=int, choices=range(1, 6), default=None,
                    metavar="{1..5}",
                    help="이해도 별점(1~5). 카메라로 측정할 수 없어 직접 입력받는다. --upload 시 필수.")
    args = ap.parse_args()
    if args.upload and not args.subject:
        ap.error("--upload 를 쓰려면 --subject 도 지정해야 합니다.")
    if args.upload and args.understanding is None:
        ap.error("--upload 를 쓰려면 --understanding(1~5)도 지정해야 합니다.")
    return args


def _to_stars(focused_ratio: float) -> int:
    """'집중' 비율(0~1)을 Backend 별점 1~5로. 0%→1, 50%→3, 100%→5."""
    return 1 + int(max(0.0, min(1.0, focused_ratio)) * 4 + 0.5)  # round()는 2.5→2라 쓰지 않는다


def _upload_session(args, counts: Counter, started_at: float) -> None:
    """세션 집계를 백엔드에 기록. 실패해도 예외를 올리지 않는다 — 측정은 이미 끝났고
    여기서 죽으면 사용자는 세션 요약조차 못 본다."""
    import os
    from pi_backend import BackendClient, BackendError

    student_id = args.student_id or (int(os.getenv("STUDENT_ID")) if os.getenv("STUDENT_ID") else None)
    if not student_id:
        print("[업로드] 학생 ID가 없습니다 — --student-id 또는 .env의 STUDENT_ID를 설정하세요.")
        return

    graded = sum(v for k, v in counts.items() if k != "얼굴 미검출")
    if graded == 0:
        print("[업로드] 얼굴이 한 번도 검출되지 않아 전송할 집계가 없습니다.")
        return

    focused_ratio = counts[FOCUSED_LABEL] / graded
    drowsy_ratio = counts["졸음"] / graded
    concentration = _to_stars(focused_ratio)
    duration_min = max(1, round((time.time() - started_at) / 60))

    try:
        BackendClient().save_learning_activity(
            student_id=student_id,
            subject=args.subject,
            study_date=time.strftime("%Y-%m-%d", time.localtime(started_at)),
            study_start_time=time.strftime("%H:%M:%S", time.localtime(started_at)),
            duration_minutes=duration_min,
            understanding_score=args.understanding,
            concentration_score=concentration,
            focused_ratio=focused_ratio,
            drowsy_ratio=drowsy_ratio,
            measurement_source="CAMERA",
        )
        print(f"[업로드] 기록 완료 — {duration_min}분, 집중 {focused_ratio * 100:.0f}% → 집중도 {concentration}/5, "
              f"이해도 {args.understanding}/5")
    except BackendError as e:
        print(f"[업로드] 실패: {e}")


def _upload_posture_session(args, session):
    if session is None or session.total == 0:
        return
    import os
    from pi_backend import BackendClient, BackendError
    from rpi.pose_main import persist_report
    student_id = args.student_id or (int(os.environ['STUDENT_ID']) if os.getenv('STUDENT_ID') else None)
    if not student_id:
        print('[자세 업로드] 학생 ID가 없습니다.')
        return
    record = session.record(student_id, args.subject)
    path = None
    try:
        path = persist_report(record)
        saved = BackendClient().save_posture_record(record)
        print(f"[자세 업로드] #{saved['id']} 저장, 보고서: {path}")
    except (BackendError, OSError, ValueError) as error:
        retry = f'; python rpi/pose_main.py --retry {path}' if path else ''
        print(f'[자세 업로드 실패] {error}{retry}')


def main():
    if '--pose-only' in sys.argv[1:]:
        from rpi.pose_main import main as pose_main
        return pose_main([argument for argument in sys.argv[1:] if argument != '--pose-only'])
    import cv2
    from rpi.live_attention import AttentionEstimator

    args = _parse_args()
    attention = AttentionEstimator()
    pose_estimator = try_load_pose_estimator()
    pose_session = PostureSession('CAMERA', pose_estimator.path.stem) if pose_estimator else None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        if pose_estimator:
            pose_estimator.close()
        raise RuntimeError("카메라를 열 수 없습니다 (index 0)")

    last_infer = 0.0
    counts: Counter = Counter()
    started_at = time.time()
    started_monotonic = time.monotonic()
    print("실행 시작 (Ctrl+C로 종료)")
    try:
        while time.monotonic() - started_monotonic < 86400:
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
            counts[status["집중도"]] += 1

            if pose_estimator:
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    people = pose_estimator.infer(rgb)
                    status['자세'] = pose_session.add(people, frame.shape[:2])
                except Exception as e:
                    print(f"[Pose] 추론 오류: {e}")
                    pose_estimator.close()
                    pose_estimator = None
                    pose_session = None

            print(status)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if pose_session:
            pose_session.finish()
        if pose_estimator:
            pose_estimator.close()
        if counts:
            total = sum(counts.values())
            print("\n=== 세션 요약 ===")
            for label, n in counts.most_common():
                print(f"  {label}: {n}회 ({n / total * 100:.0f}%)")
            print(f"  측정 시간: {(time.time() - started_at) / 60:.1f}분")
        if args.upload:
            _upload_session(args, counts, started_at)
            _upload_posture_session(args, pose_session)


if __name__ == "__main__":
    raise SystemExit(main())
