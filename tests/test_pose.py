import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from pi_backend import BackendClient, BackendError
from rpi.pose_hailo import decode_detections, letterbox
from rpi.pose_main import crop_frame, persist_report
from rpi.posture_heuristic import classify_posture
from rpi.posture_session import PostureSession


def heads():
    return {f"{size}-{channels}": np.zeros((1, size, size, channels), np.float32)
            for size in (80, 40, 20) for channels in (64, 1, 51)}


def candidate(outputs, row, column, score, distance=2):
    outputs['80-1'][0, row, column, 0] = score
    distribution = np.full((4, 16), -100, np.float32)
    distribution[:, distance] = 100
    outputs['80-64'][0, row, column] = distribution.ravel()
    points = np.zeros((17, 3), np.float32)
    points[:, 0] = 1
    points[:, 1] = 2
    outputs['80-51'][0, row, column] = points.ravel()


def person(nose_y=80, right_y=120):
    points = [(300, 100, 0.9)] * 17
    points[0] = (300, nose_y, 0.9)
    points[5] = (250, 120, 0.9)
    points[6] = (350, right_y, 0.9)
    return points


class DecoderTest(unittest.TestCase):
    def test_empty_frame(self):
        self.assertEqual(decode_detections(heads()), [])

    def test_grid_dfl_and_all_coco_joints(self):
        outputs = heads()
        candidate(outputs, 2, 1, 0.9)
        result, = decode_detections(outputs)
        np.testing.assert_allclose(result.box, (-4, 4, 28, 36), atol=1e-5)
        self.assertEqual(len(result.keypoints), 17)
        np.testing.assert_allclose(result.keypoints[0], (24, 48, 0.5), atol=1e-5)
        np.testing.assert_allclose(result.keypoints[16], result.keypoints[0])

    def test_nms_retains_highest_scoring_box_independent_of_tensor_order(self):
        outputs = heads()
        candidate(outputs, 10, 10, 0.7, distance=4)
        candidate(outputs, 10, 11, 0.95, distance=4)
        result = decode_detections(dict(reversed(list(outputs.items()))))
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].score, 0.95, places=5)

    def test_separate_people_are_preserved(self):
        outputs = heads()
        candidate(outputs, 10, 10, 0.9)
        candidate(outputs, 50, 50, 0.8)
        self.assertEqual(len(decode_detections(outputs)), 2)

    def test_missing_duplicate_invalid_or_quantized_heads_are_rejected(self):
        for change in ('missing', 'duplicate', 'nan', 'uint8', 'score'):
            outputs = heads()
            if change == 'missing': outputs.pop('20-64')
            if change == 'duplicate': outputs['duplicate'] = outputs['20-64']
            if change == 'nan': outputs['20-64'][0, 0, 0, 0] = np.nan
            if change == 'uint8': outputs['20-64'] = outputs['20-64'].astype(np.uint8)
            if change == 'score': outputs['20-1'][0, 0, 0, 0] = 2
            with self.subTest(change=change), self.assertRaises(ValueError):
                decode_detections(outputs)

    def test_rectangular_letterbox_preserves_geometry_and_rgb(self):
        image = np.full((240, 640, 3), (30, 60, 90), np.uint8)
        boxed, transform = letterbox(image, (640, 640))
        self.assertEqual(transform, (1, 1, 0, 200))
        np.testing.assert_array_equal(boxed[0, 0], (114, 114, 114))
        np.testing.assert_array_equal(boxed[200, 0], (30, 60, 90))

    def test_fixed_student_area_crops_only_the_configured_region(self):
        image = np.arange(100 * 200 * 3).reshape(100, 200, 3)
        np.testing.assert_array_equal(crop_frame(image, (0.25, 0.1, 0.5, 0.4)), image[10:50, 50:150])

    def test_record_is_frozen_for_retries_and_cannot_accept_later_samples(self):
        session = PostureSession('VIDEO', submission_id='test')
        session.add([person()])
        first = session.record(2, '수학')
        with patch('rpi.posture_session.time.monotonic', return_value=999999999):
            self.assertEqual(session.record(2, '수학'), first)
        with self.assertRaises(RuntimeError): session.add([person()])


class PostureTest(unittest.TestCase):
    def test_four_posture_states(self):
        cases = [(person(), '바른 자세'), (person(128), '구부정함'),
                 (person(160), '엎드림'), (person(80, 160), '기울어짐')]
        for points, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(classify_posture(points, image_shape=(480, 640)).label, expected)

    def test_low_confidence_offscreen_degenerate_and_nonfinite_are_unknown(self):
        for value in ((300, 80, 0.1), (-1, 80, 0.9), (math.nan, 80, 0.9), (640, 80, 0.9)):
            points = person(); points[0] = value
            self.assertEqual(classify_posture(points, image_shape=(480, 640)).label, '판단불가')
        points = person(); points[6] = points[5]
        self.assertEqual(classify_posture(points).label, '판단불가')
        self.assertEqual(classify_posture([]).label, '판단불가')

    def test_missing_and_multiple_people_never_become_good_posture(self):
        session = PostureSession('IMAGE', submission_id='test')
        session.add([])
        session.add([person(), person()])
        session.add([person()])
        result = session.record(2, '수학')
        self.assertEqual((result['totalSamples'], result['noPersonSamples'], result['multiplePeopleSamples'], result['uprightSamples']), (3, 1, 1, 1))
        self.assertEqual(result['inputSource'], 'IMAGE')
        self.assertNotIn('concentrationScore', result)

    def test_failed_upload_can_retry_identical_saved_report(self):
        session = PostureSession('IMAGE', submission_id='test')
        session.add([person()])
        record = session.record(2, '수학')
        with tempfile.TemporaryDirectory() as directory:
            path = persist_report(record, Path(directory) / 'record.json')
            import json
            self.assertEqual(json.loads(path.read_text(encoding='utf-8')), record)
            with self.assertRaises(FileExistsError): persist_report(record, path)
        with self.assertRaises(ValueError): persist_report({**record, 'submissionId': '../../unsafe'})

    @patch.object(BackendClient, '_request', return_value={'id': 1, 'studentId': 2, 'submissionId': 'test'})
    def test_posture_upload_uses_separate_endpoint_and_rejects_bad_counters(self, request):
        session = PostureSession('IMAGE', submission_id='test'); session.add([person()])
        record = session.record(2, '수학')
        client = BackendClient(base_url='http://backend.test', token='test')
        client.save_posture_record(record)
        self.assertEqual(request.call_args.args, ('POST', '/api/posture-records'))
        for change in ({'totalSamples': 2}, {'uprightSamples': True}, {'inputSource': 'OTHER'}, {'durationSeconds': 0}):
            with self.subTest(change=change), self.assertRaises(BackendError):
                client.save_posture_record({**record, **change})


if __name__ == '__main__':
    unittest.main()
