"""Aggregate posture separately from face-based attention and self-evaluation."""
from collections import Counter
from datetime import datetime, timezone
import math
import time
import uuid

from rpi.posture_heuristic import classify_posture

COUNT_FIELDS = {"바른 자세": "uprightSamples", "구부정함": "slouchedSamples",
                "엎드림": "headDownSamples", "기울어짐": "tiltedSamples",
                "판단불가": "unknownSamples", "사람 미검출": "noPersonSamples",
                "여러 명 검출": "multiplePeopleSamples"}


class PostureSession:
    def __init__(self, input_source="CAMERA", model_name="yolov8s_pose_h8", submission_id=None):
        self.input_source = input_source
        self.model_name = model_name
        self.submission_id = submission_id or str(uuid.uuid4())
        self.started_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        self.started_monotonic = time.monotonic()
        self.finished_monotonic = None
        self.counts = Counter()

    def add(self, people, image_shape=None):
        if self.finished_monotonic is not None:
            raise RuntimeError("Posture session has already ended")
        if not people:
            label = "사람 미검출"
        elif len(people) > 1:
            label = "여러 명 검출"
        else:
            label = classify_posture(people[0], image_shape=image_shape).label
        self.counts[label] += 1
        return label

    @property
    def total(self):
        return sum(self.counts.values())

    def record(self, student_id, subject):
        self.finish()
        return {"studentId": student_id, "subject": subject, "submissionId": self.submission_id,
                "startedAt": self.started_at, "durationSeconds": min(86400, max(1, math.ceil(self.finished_monotonic - self.started_monotonic))),
                "inputSource": self.input_source, "modelName": self.model_name,
                "totalSamples": self.total,
                **{field: self.counts[label] for label, field in COUNT_FIELDS.items()}}

    def finish(self):
        if self.finished_monotonic is None:
            self.finished_monotonic = time.monotonic()
