"""Run AI HAT posture analysis without a face model or self-evaluation scores."""
import argparse
import json
import os
import re
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pi_backend import BackendClient, BackendError
from rpi.pose_hailo import HailoPoseEstimator
from rpi.posture_session import PostureSession


def persist_report(record, output=None):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,63}", record["submissionId"]):
        raise ValueError("Invalid submission ID")
    destination = Path(output) if output else Path.home() / ".local/state/i-route/posture" / (record["submissionId"] + ".json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        os.chmod(destination, 0o600)
        json.dump(record, stream, ensure_ascii=False, indent=2)
    return destination


def crop_frame(frame, roi):
    if roi is None:
        return frame
    x, y, width, height = roi
    h, w = frame.shape[:2]
    left, top, right, bottom = round(x * w), round(y * h), round((x + width) * w), round((y + height) * h)
    if right <= left or bottom <= top:
        raise ValueError("Measurement area is too small")
    return frame[top:bottom, left:right]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="camera", help="camera, an image, or a video file")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--hef", type=Path)
    parser.add_argument("--interval", type=float, default=1.0, help="Camera inference interval in seconds")
    parser.add_argument("--seconds", type=float, help="Stop a camera session after this many seconds")
    parser.add_argument("--max-samples", type=int, default=86400)
    parser.add_argument("--roi", help="Optional fixed student area: x,y,width,height, each normalized to the image")
    parser.add_argument("--student-id", type=int, default=None)
    parser.add_argument("--subject", default=None)
    parser.add_argument("--submission-id")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--retry", type=Path, help="Resend a saved JSON report without measuring again")
    args = parser.parse_args(argv)
    roi = None
    if args.roi:
        try:
            roi = tuple(map(float, args.roi.split(',')))
            x, y, width, height = roi
            if not (0 <= x < 1 and 0 <= y < 1 and width > 0 and height > 0 and x + width <= 1 and y + height <= 1):
                raise ValueError()
        except (ValueError, TypeError):
            parser.error("--roi must be a non-empty normalized area inside the image")
    if args.retry:
        try:
            record = json.loads(args.retry.read_text(encoding="utf-8"))
            if not isinstance(record, dict):
                raise ValueError('Saved posture report must be an object')
            if args.student_id is not None and args.student_id != record.get('studentId'):
                raise ValueError('A retry must keep the original student ID')
            if args.subject is not None and args.subject != record.get('subject'):
                raise ValueError('A retry must keep the original subject')
            saved = BackendClient().save_posture_record(record)
            print(f"[자세 저장] #{saved['id']} (동일 제출 ID 재전송)")
            return 0
        except (OSError, ValueError, BackendError) as error:
            print(f"[자세 저장 실패] {error}", file=sys.stderr)
            return 1
    try:
        student_id = args.student_id if args.student_id is not None else (int(os.environ["STUDENT_ID"]) if os.getenv("STUDENT_ID") else None)
    except ValueError:
        parser.error("STUDENT_ID must be a positive integer")
    if args.subject is not None:
        args.subject = args.subject.strip()
        if len(args.subject) > 255:
            parser.error("--subject is too long")
    if args.upload and (not student_id or not args.subject):
        parser.error("--upload requires a student ID and --subject")
    if student_id is not None and student_id <= 0:
        parser.error("--student-id must be positive")
    if args.submission_id and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,63}", args.submission_id):
        parser.error("Invalid submission ID")
    if not 0.05 <= args.interval <= 60 or not 1 <= args.max_samples <= 86400:
        parser.error("--interval must be 0.05..60 and --max-samples must be 1..86400")
    if args.seconds is not None and not 0 < args.seconds <= 86400:
        parser.error("--seconds must be greater than 0 and at most 86400")
    import cv2
    source = "CAMERA" if args.input == "camera" else ("IMAGE" if Path(args.input).suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp") else "VIDEO")
    capture = None
    session = None
    try:
        with HailoPoseEstimator(args.hef) as estimator:
            session = PostureSession(source, estimator.path.stem, args.submission_id)
            print(f"[자세] {estimator.path.name}, 입력={source}; Ctrl+C로 종료", flush=True)
            if source == "IMAGE":
                frame = cv2.imread(args.input)
                if frame is None:
                    raise ValueError("Cannot read the input image")
                frame = crop_frame(frame, roi)
                people = estimator.infer(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                print({"자세": session.add(people, frame.shape[:2]), "사람 수": len(people)}, flush=True)
            else:
                capture = cv2.VideoCapture(args.camera_index, cv2.CAP_V4L2) if source == "CAMERA" else cv2.VideoCapture(args.input)
                if not capture.isOpened():
                    raise RuntimeError("Cannot open posture input")
                last_infer = float("-inf")
                try:
                    while session.total < args.max_samples:
                        now = time.monotonic()
                        if now - session.started_monotonic >= (args.seconds or 86400):
                            break
                        ok, frame = capture.read()
                        if not ok:
                            if source == "VIDEO":
                                break
                            raise RuntimeError("Camera stopped delivering frames")
                        if source == "CAMERA" and now - last_infer < args.interval:
                            continue
                        last_infer = now
                        frame = crop_frame(frame, roi)
                        people = estimator.infer(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        print({"자세": session.add(people, frame.shape[:2]), "사람 수": len(people)}, flush=True)
                except KeyboardInterrupt:
                    pass
    except (Exception, KeyboardInterrupt) as error:
        print(f"[자세 분석 실패] {error}", file=sys.stderr)
        return 1
    finally:
        if capture is not None:
            capture.release()
    if session is None or not session.total:
        print("[자세] 전송할 표본이 없습니다.")
        return 0
    record = session.record(student_id, args.subject or "수학")
    try:
        path = persist_report(record, args.output)
    except (OSError, ValueError) as error:
        print(f"[보고서 저장 실패] {error}", file=sys.stderr)
        return 1
    print(f"[자세 집계] {json.dumps(record, ensure_ascii=False)}", flush=True)
    print(f"[자세 보고서] {path}", flush=True)
    if args.upload:
        try:
            saved = BackendClient().save_posture_record(record)
            print(f"[자세 저장] #{saved['id']}", flush=True)
        except BackendError as error:
            print(f"[자세 저장 실패] {error}; --retry {path} 로 같은 결과를 재전송하세요.", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
