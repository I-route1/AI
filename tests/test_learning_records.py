import io
import json
import math
import unittest
import urllib.error
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

from pi_backend import BackendClient, BackendError
from ocr.pi_ocr_grade import _upload_report
from rpi.pi_main import _upload_session


class BackendClientTest(unittest.TestCase):
    def setUp(self):
        self.client = BackendClient(base_url="http://backend.test", token="test-jwt")

    def activity(self, **kwargs):
        return self.client.save_learning_activity(14, "수학", "2026-11-04", "10:00:00", 30, 4, 3, **kwargs)

    @patch("urllib.request.urlopen")
    def test_ocr_retry_preserves_submission_and_unicode(self, urlopen):
        urlopen.side_effect = [urllib.error.URLError("timeout"), io.BytesIO(b'{"id":1}')]
        questions = [{"questionId": "Q1", "expectedAnswer": "돼지", "recognizedText": "되지",
                      "similarity": 0.5, "correct": False, "conceptTag": "받침", "errorType": None}]
        with patch("pi_backend.time.sleep"):
            saved = self.client.save_ocr_result(14, "국어", "same-id", 1, 0, questions)
        self.assertEqual(saved, {"id": 1})
        requests = [call.args[0] for call in urlopen.call_args_list]
        self.assertEqual(requests[0].data, requests[1].data)
        payload = json.loads(requests[0].data)
        self.assertEqual(payload["submissionId"], "same-id")
        self.assertEqual(payload["questions"], questions)
        self.assertEqual(requests[0].get_header("Authorization"), "Bearer test-jwt")

    @patch.object(BackendClient, "_request", return_value=[])
    def test_read_apis(self, request):
        self.client.get_wrong_answers(14)
        request.assert_called_with("GET", "/api/wrong-answer/14", params=None)
        self.client.get_wrong_answers(14, "국어")
        request.assert_called_with("GET", "/api/wrong-answer/14", params={"subject": "국어"})
        self.client.get_ocr_results(14, "국어")
        request.assert_called_with("GET", "/api/ocr-results", params={"studentId": 14, "subject": "국어"})
        self.client.get_ocr_result(1, 14)
        request.assert_called_with("GET", "/api/ocr-results/1", params={"studentId": 14})

    @patch.object(BackendClient, "_request", return_value={})
    def test_camera_details_and_manual_default(self, request):
        self.activity(focused_ratio=0.6, drowsy_ratio=0.2, measurement_source="CAMERA")
        payload = json.loads(request.call_args.kwargs["body"])
        self.assertEqual((payload["concentrationScore"], payload["focusedRatio"], payload["drowsyRatio"],
                          payload["measurementSource"]), (3, 0.6, 0.2, "CAMERA"))
        self.activity()
        payload = json.loads(request.call_args.kwargs["body"])
        self.assertEqual(payload["measurementSource"], "MANUAL")
        self.assertIsNone(payload["focusedRatio"])
        self.assertIsNone(payload["drowsyRatio"])

    @patch.object(BackendClient, "_request")
    def test_rejects_bad_ratios_before_network(self, request):
        invalid = [dict(focused_ratio=-0.1), dict(drowsy_ratio=1.1), dict(focused_ratio=math.nan),
                   dict(drowsy_ratio=math.inf), dict(focused_ratio=0.8, drowsy_ratio=0.3),
                   dict(measurement_source="CAMERA"), dict(measurement_source="CAMERA", focused_ratio=1),
                   dict(measurement_source="OTHER")]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(BackendError):
                self.activity(**values)
        request.assert_not_called()


class PiUploadTest(unittest.TestCase):
    def args(self):
        return SimpleNamespace(student_id=14, subject="국어", submission_id="scan-1", understanding=4)

    @patch("pi_backend.BackendClient")
    def test_perfect_score_still_uploads_all_recognized_text(self, client):
        client.return_value.save_ocr_result.return_value = {"id": 7}
        report = SimpleNamespace(total=1, score=1, results=[SimpleNamespace(
            question_id="1", expected="수박", recognized="수박", similarity=1.0, is_correct=True)])
        _upload_report(report, [{"question_id": "1", "concept_tag": "명사"}], self.args())
        args = client.return_value.save_ocr_result.call_args.args
        self.assertEqual(args[:5], (14, "국어", "scan-1", 1, 1))
        self.assertEqual(args[5][0]["recognizedText"], "수박")
        self.assertTrue(args[5][0]["correct"])
        client.return_value.record_wrong_answer.assert_not_called()

    @patch("pi_backend.BackendClient")
    def test_uploads_wrong_and_correct_questions_together(self, client):
        client.return_value.save_ocr_result.return_value = {"id": 8}
        report = SimpleNamespace(total=2, score=1, results=[
            SimpleNamespace(question_id="1", expected="돼지", recognized="되지", similarity=0.5, is_correct=False),
            SimpleNamespace(question_id="2", expected="수박", recognized="수박", similarity=1.0, is_correct=True)])
        _upload_report(report, [{"question_id": "1", "concept_tag": "받침", "error_type": "MEMORIZATION_GAP"}], self.args())
        questions = client.return_value.save_ocr_result.call_args.args[5]
        self.assertEqual(len(questions), 2)
        self.assertEqual(questions[0]["errorType"], "MEMORIZATION_GAP")
        self.assertEqual(questions[1]["conceptTag"], "국어")
        client.return_value.record_wrong_answer.assert_not_called()

    @patch("pi_backend.BackendClient")
    def test_no_face_samples_are_excluded_from_ratios(self, client):
        counts = Counter({"집중": 6, "집중하지않음": 2, "졸음": 2, "얼굴 미검출": 10})
        with patch("rpi.pi_main.time.time", return_value=1800):
            _upload_session(self.args(), counts, 0)
        kwargs = client.return_value.save_learning_activity.call_args.kwargs
        self.assertEqual(kwargs["focused_ratio"], 0.6)
        self.assertEqual(kwargs["drowsy_ratio"], 0.2)
        self.assertEqual(kwargs["measurement_source"], "CAMERA")
        self.assertEqual(kwargs["concentration_score"], 3)

    @patch("pi_backend.BackendClient")
    def test_does_not_upload_session_without_face_detection(self, client):
        _upload_session(self.args(), Counter({"얼굴 미검출": 10}), 0)
        client.assert_not_called()

    @patch("pi_backend.BackendClient")
    def test_failed_upload_preserves_local_grading(self, client):
        client.return_value.save_ocr_result.side_effect = BackendError("offline")
        report = SimpleNamespace(total=1, score=0, results=[SimpleNamespace(
            question_id="1", expected="돼지", recognized="", similarity=0, is_correct=False)])
        with patch("builtins.print") as log:
            _upload_report(report, [], self.args())
        self.assertIn("실패", log.call_args.args[0])
        client.return_value.record_wrong_answer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
