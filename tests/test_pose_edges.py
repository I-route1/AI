import io
import json
import os
from pathlib import Path
import stat
import tempfile
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from pi_backend import BackendClient, BackendError
from rpi.pose_hailo import HailoPoseEstimator, decode_detections, letterbox
from rpi.pose_main import main, persist_report
from rpi.posture_session import PostureSession


def record():
    session = PostureSession('VIDEO', submission_id='edge-test')
    session.add([])
    return session.record(2, '수학')


class FileTest(unittest.TestCase):
    def test_failed_flush_never_publishes_a_partial_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            with patch('rpi.pose_main.os.fsync', side_effect=OSError('disk full')):
                with self.assertRaises(OSError): persist_report(record(), path)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_concurrent_writers_publish_one_complete_private_report_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            payloads = [record(), {**record(), 'subject': '국어'}]
            def write(payload):
                try: persist_report(payload, path); return True
                except FileExistsError: return False
            with ThreadPoolExecutor(max_workers=2) as pool:
                self.assertEqual(sorted(pool.map(write, payloads)), [False, True])
            self.assertIn(json.loads(path.read_text(encoding='utf-8')), payloads)
            self.assertEqual(list(Path(directory).iterdir()), [path])
            if os.name != 'nt': self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_invalid_json_is_not_written_and_existing_report_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            persist_report(record(), path)
            before = path.read_bytes()
            with self.assertRaises(ValueError): persist_report({**record(), 'durationSeconds': float('nan')}, path)
            self.assertEqual(path.read_bytes(), before)


class ClientTest(unittest.TestCase):
    def test_invalid_metadata_is_rejected_before_network(self):
        client = BackendClient(base_url='http://backend.test', token='test')
        changes = [{'studentId': 2**63}, {'subject': 123}, {'subject': '\u200b\u00a0'}, {'subject': '😀' * 128},
                   {'subject': '수\ud800'}, {'submissionId': 123}, {'modelName': 'model name'},
                   {'startedAt': '2026-11-11'}, {'startedAt': 'bad'}, {'startedAt': '1000-01-01T00:00:00+18:00'}]
        with patch.object(client, '_request') as request:
            for change in changes:
                with self.subTest(change=repr(change)), self.assertRaises(BackendError):
                    client.save_posture_record({**record(), **change})
            request.assert_not_called()

    def test_wrong_acknowledgement_never_reports_success(self):
        client = BackendClient(base_url='http://backend.test', token='test')
        for response in ({'id': 1}, {'id': 1, 'studentId': 9, 'submissionId': 'edge-test'},
                         {'id': 1, 'studentId': 2, 'submissionId': 'other'}, {'id': True}, None):
            with self.subTest(response=response), patch.object(client, '_request', return_value=response), self.assertRaises(BackendError):
                client.save_posture_record(record())


class CliTest(unittest.TestCase):
    def run_video(self, directory, reads, inferences=None, upload=False):
        source = Path(directory) / 'input.mp4'; source.write_bytes(b'test-fixture')
        output = Path(directory) / 'output with spaces.json'
        capture = MagicMock(); capture.isOpened.return_value = True; capture.read.side_effect = reads
        estimator = MagicMock(); estimator.path = Path('yolov8s_pose_h8.hef')
        estimator.infer.side_effect = inferences or [[], []]
        with patch('rpi.pose_main.HailoPoseEstimator') as factory, patch('cv2.VideoCapture', return_value=capture), \
                patch('rpi.pose_main.BackendClient') as backend, patch('sys.stdout', new_callable=io.StringIO), patch('sys.stderr', new_callable=io.StringIO):
            factory.return_value.__enter__.return_value = estimator
            backend.return_value.save_posture_record.side_effect = BackendError('offline') if upload else None
            arguments = ['--input', str(source), '--student-id', '2', '--subject', '수학', '--output', str(output)]
            result = main(arguments + (['--upload'] if upload else []))
            factory.return_value.__exit__.assert_called_once()
            capture.release.assert_called_once()
            return result, output, backend

    def test_empty_or_unreadable_video_is_failure_without_a_record(self):
        with tempfile.TemporaryDirectory() as directory:
            result, path, backend = self.run_video(directory, [(False, None)])
            self.assertEqual(result, 1); self.assertFalse(path.exists()); backend.assert_not_called()

    def test_end_of_file_and_keyboard_stop_preserve_valid_samples(self):
        for end in ((False, None), KeyboardInterrupt()):
            with self.subTest(end=type(end).__name__), tempfile.TemporaryDirectory() as directory:
                result, path, _ = self.run_video(directory, [(True, np.zeros((32, 32, 3), np.uint8)), end])
                self.assertEqual(result, 0)
                self.assertEqual(json.loads(path.read_text())['noPersonSamples'], 1)

    def test_npu_failure_does_not_upload_a_partial_session(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = (True, np.zeros((32, 32, 3), np.uint8))
            result, path, backend = self.run_video(directory, [frame, frame], [[], RuntimeError('NPU timeout')])
            self.assertEqual(result, 1); self.assertFalse(path.exists()); backend.assert_not_called()

    def test_disconnected_camera_releases_capture_without_uploading(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = MagicMock(); capture.isOpened.return_value = True; capture.read.return_value = (False, None)
            with patch('rpi.pose_main.HailoPoseEstimator') as factory, patch('cv2.VideoCapture', return_value=capture), \
                    patch('rpi.pose_main.BackendClient') as backend, patch('sys.stdout', new_callable=io.StringIO), patch('sys.stderr', new_callable=io.StringIO):
                factory.return_value.__enter__.return_value.path = Path('yolov8s_pose_h8.hef')
                path = Path(directory) / 'report.json'
                self.assertEqual(main(['--seconds', '1', '--output', str(path)]), 1)
                capture.release.assert_called_once(); factory.return_value.__exit__.assert_called_once()
                self.assertFalse(path.exists()); backend.assert_not_called()

    def test_upload_failure_keeps_report_and_retry_preserves_original_content(self):
        with tempfile.TemporaryDirectory() as directory:
            result, path, _ = self.run_video(directory, [(True, np.zeros((32, 32, 3), np.uint8)), (False, None)], upload=True)
            self.assertEqual(result, 1)
            before = path.read_bytes()
            with patch('rpi.pose_main.BackendClient') as backend, patch('sys.stdout', new_callable=io.StringIO):
                backend.return_value.save_posture_record.return_value = {'id': 1}
                self.assertEqual(main(['--retry', str(path)]), 0)
                self.assertEqual(backend.return_value.save_posture_record.call_args.args[0], json.loads(before))
                self.assertEqual(path.read_bytes(), before)
            for changes in (['--student-id', '9'], ['--subject', '국어'], ['--submission-id', 'other']):
                with patch('rpi.pose_main.BackendClient') as backend, patch('sys.stderr', new_callable=io.StringIO):
                    self.assertEqual(main(['--retry', str(path), *changes]), 1)
                    backend.assert_not_called()

    def test_invalid_arguments_do_not_open_the_npu_or_camera(self):
        for arguments in (['--roi', '0,0,0,1'], ['--seconds', 'nan'], ['--interval', 'inf'], ['--camera-index', '-1'],
                          ['--max-samples', '86401'], ['--subject', '  '], ['--input', 'missing.mp4']):
            with self.subTest(arguments=arguments), patch('rpi.pose_main.HailoPoseEstimator') as estimator, \
                    patch('sys.stderr', new_callable=io.StringIO), self.assertRaises(SystemExit):
                try: main(arguments)
                finally: estimator.assert_not_called()


class RuntimeTest(unittest.TestCase):
    def module(self):
        outputs = {f'{size}-{channels}': np.zeros((size, size, channels), np.float32)
                   for size in (80, 40, 20) for channels in (64, 1, 51)}
        hef = MagicMock(); hef.get_input_vstream_infos.return_value = [SimpleNamespace(name='input', shape=(640, 640, 3))]
        hef.get_output_vstream_infos.return_value = [SimpleNamespace(name=name, shape=value.shape) for name, value in outputs.items()]
        device = MagicMock(); network = MagicMock(); device.__enter__.return_value.configure.return_value = [network]
        pipeline = MagicMock(); pipeline.__enter__.return_value.infer.return_value = outputs
        module = SimpleNamespace(HEF=MagicMock(return_value=hef), VDevice=MagicMock(return_value=device),
                ConfigureParams=MagicMock(), HailoStreamInterface=SimpleNamespace(PCIe='pcie'),
                InputVStreamParams=MagicMock(), OutputVStreamParams=MagicMock(),
                FormatType=SimpleNamespace(UINT8='uint8', FLOAT32='float32'), InferVStreams=MagicMock(return_value=pipeline))
        return module, device, network, pipeline

    def test_context_failure_releases_device_and_activation(self):
        module, device, network, pipeline = self.module()
        pipeline.__enter__.side_effect = RuntimeError('busy')
        with tempfile.NamedTemporaryFile(suffix='.hef') as file, patch.dict('sys.modules', {'hailo_platform': module}):
            with self.assertRaises(RuntimeError): HailoPoseEstimator(file.name)
        device.__exit__.assert_called_once(); network.activate.return_value.__exit__.assert_called_once()

    def test_inference_exception_releases_all_resources_and_repeated_close_is_safe(self):
        module, device, network, pipeline = self.module()
        pipeline.__enter__.return_value.infer.side_effect = RuntimeError('timeout')
        with tempfile.NamedTemporaryFile(suffix='.hef') as file, patch.dict('sys.modules', {'hailo_platform': module}):
            estimator = HailoPoseEstimator(file.name)
            with self.assertRaises(RuntimeError):
                with estimator: estimator.infer(np.zeros((240, 320, 3), np.uint8))
            estimator.close()
        device.__exit__.assert_called_once(); pipeline.__exit__.assert_called_once()

    def test_invalid_geometry_and_empty_inputs_are_rejected(self):
        for shape in ((0, 640), (641, 640), (640,), (640, 640.0)):
            with self.subTest(shape=shape), self.assertRaises(ValueError): decode_detections({}, shape)
        for frame in (np.zeros((0, 4, 3), np.uint8), np.zeros((4, 4), np.uint8), np.zeros((4, 4, 3), np.float32)):
            with self.assertRaises(ValueError): letterbox(frame, (640, 640))


if __name__ == '__main__':
    unittest.main()
