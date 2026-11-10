"""Hailo YOLOv8 COCO-17 inference, letterboxing and verified raw-head decoding.

DFL/grid conventions follow the official Hailo TAPPAS pose implementation:
https://github.com/hailo-ai/hailo-apps/blob/main/hailo_apps/postprocess/cpp/yolov8pose_postprocess.cpp
Only the three-level 64/1/51-channel layout is accepted.
"""
from contextlib import ExitStack
from dataclasses import dataclass
import os
from pathlib import Path
import numpy as np


@dataclass
class PoseDetection:
    box: tuple[float, float, float, float]
    score: float
    keypoints: list[tuple[float, float, float]]


def default_hef_path():
    if os.getenv("POSE_HEF_PATH"):
        return Path(os.environ["POSE_HEF_PATH"]).expanduser()
    local = Path(__file__).with_name("yolov8_pose.hef")
    return local if local.is_file() else Path("/usr/share/hailo-models/yolov8s_pose_h8.hef")


def letterbox(frame_rgb, size):
    import cv2
    if frame_rgb.dtype != np.uint8 or frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
        raise ValueError("Pose input must be a non-empty RGB uint8 image")
    height, width = frame_rgb.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Pose input must be a non-empty RGB uint8 image")
    target_h, target_w = size
    scale = min(target_w / width, target_h / height)
    resized_w, resized_h = max(1, round(width * scale)), max(1, round(height * scale))
    left, top = (target_w - resized_w) // 2, (target_h - resized_h) // 2
    image = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    image[top:top + resized_h, left:left + resized_w] = cv2.resize(frame_rgb, (resized_w, resized_h))
    return image, (resized_w / width, resized_h / height, left, top)


def decode_detections(outputs, input_shape=(640, 640), *, score_threshold=0.6,
                      iou_threshold=0.7, max_detections=100):
    """Decode dequantized HWC arrays while retaining all 17 joints in COCO order."""
    if not 0 < score_threshold <= 1 or not 0 < iou_threshold <= 1 or type(max_detections) is not int or max_detections < 1:
        raise ValueError("Invalid pose thresholds")
    if len(input_shape) != 2 or any(type(value) is not int or value <= 0 or value % 32 for value in input_shape):
        raise ValueError("Pose dimensions must be positive multiples of 32")
    levels = {}
    for name, value in outputs.items():
        array = np.asarray(value)
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 3 or array.shape[-1] not in (64, 1, 51):
            raise ValueError(f"Unsupported pose tensor {name}: {array.shape}")
        if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
            raise ValueError(f"Pose tensor must be finite and dequantized: {name}")
        heads = levels.setdefault(array.shape[:2], {})
        if array.shape[-1] in heads:
            raise ValueError(f"Duplicate pose tensor head: {name}")
        heads[array.shape[-1]] = array
    height, width = input_shape
    expected = {(height // s, width // s) for s in (8, 16, 32)}
    if len(outputs) != 9 or set(levels) != expected or any(set(h) != {64, 1, 51} for h in levels.values()):
        raise ValueError("Expected raw 64/1/51-channel pose levels at strides 8,16,32")
    boxes, scores, joints = [], [], []
    for (grid_h, grid_w), heads in levels.items():
        confidence = heads[1][..., 0]
        if confidence.min() < -1e-5 or confidence.max() > 1.00001:
            raise ValueError("Class scores must already be sigmoid probabilities")
        rows, columns = np.nonzero(confidence >= score_threshold)
        if not len(rows):
            continue
        distribution = heads[64][rows, columns].reshape(-1, 4, 16)
        distribution = distribution - distribution.max(axis=-1, keepdims=True)
        weights = np.exp(distribution)
        weights /= weights.sum(axis=-1, keepdims=True)
        stride = width / grid_w
        distances = (weights @ np.arange(16, dtype=np.float32)) * stride
        centers = np.column_stack((columns + 0.5, rows + 0.5)) * stride
        boxes.append(np.column_stack((centers - distances[:, :2], centers + distances[:, 2:])))
        points = heads[51][rows, columns].reshape(-1, 17, 3).copy()
        points[..., :2] = (points[..., :2] * 2 + np.column_stack((columns, rows))[:, None, :]) * stride
        points[..., 2] = 1 / (1 + np.exp(-np.clip(points[..., 2], -80, 80)))
        scores.append(confidence[rows, columns])
        joints.append(points)
    if not boxes:
        return []
    boxes, scores, joints = np.concatenate(boxes), np.concatenate(scores), np.concatenate(joints)
    order = np.argsort(-scores, kind="stable")[:1000]
    selected = []
    while len(order) and len(selected) < max_detections:
        index = order[0]
        selected.append(PoseDetection(tuple(map(float, boxes[index])), float(scores[index]),
                                      [tuple(map(float, p)) for p in joints[index]]))
        remaining = order[1:]
        if not len(remaining):
            break
        overlap = np.maximum(0, np.minimum(boxes[index, 2:], boxes[remaining, 2:])
                             - np.maximum(boxes[index, :2], boxes[remaining, :2]))
        intersection = overlap[:, 0] * overlap[:, 1]
        area = np.prod(np.maximum(0, boxes[index, 2:] - boxes[index, :2]))
        other_area = np.prod(np.maximum(0, boxes[remaining, 2:] - boxes[remaining, :2]), axis=1)
        iou = intersection / np.maximum(area + other_area - intersection, 1e-12)
        order = remaining[iou < iou_threshold]
    return selected


class HailoPoseEstimator:
    def __init__(self, hef_path=None):
        from hailo_platform import (HEF, VDevice, HailoStreamInterface, ConfigureParams,
                                    InputVStreamParams, OutputVStreamParams, FormatType, InferVStreams)
        self.path = Path(hef_path or default_hef_path())
        if not self.path.is_file():
            raise FileNotFoundError(f"Pose HEF not found: {self.path}")
        self._resources = ExitStack()
        try:
            self.hef = HEF(str(self.path))
            self.target = self._resources.enter_context(VDevice())
            config = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
            self.network_group = self.target.configure(self.hef, config)[0]
            input_infos = self.hef.get_input_vstream_infos()
            if len(input_infos) != 1 or len(input_infos[0].shape) != 3 or input_infos[0].shape[-1] != 3:
                raise ValueError("Expected a single RGB pose input")
            self.input_info = input_infos[0]
            self.output_infos = self.hef.get_output_vstream_infos()
            self._input_shape = tuple(self.input_info.shape)
            decode_detections({i.name: np.zeros(i.shape, dtype=np.float32) for i in self.output_infos}, self._input_shape[:2])
            inputs = InputVStreamParams.make(self.network_group, quantized=True, format_type=FormatType.UINT8, timeout_ms=10000)
            outputs = OutputVStreamParams.make(self.network_group, quantized=False, format_type=FormatType.FLOAT32, timeout_ms=10000)
            self._resources.enter_context(self.network_group.activate(self.network_group.create_params()))
            self._pipeline = self._resources.enter_context(InferVStreams(self.network_group, inputs, outputs))
        except BaseException:
            self._resources.close()
            raise

    def infer_detections(self, frame_rgb):
        image, (scale_x, scale_y, left, top) = letterbox(frame_rgb, self._input_shape[:2])
        outputs = self._pipeline.infer({self.input_info.name: image[None]})
        detections = decode_detections(outputs, self._input_shape[:2])
        for detection in detections:
            x0, y0, x1, y1 = detection.box
            detection.box = ((x0 - left) / scale_x, (y0 - top) / scale_y,
                             (x1 - left) / scale_x, (y1 - top) / scale_y)
            detection.keypoints = [((x - left) / scale_x, (y - top) / scale_y, score)
                                   for x, y, score in detection.keypoints]
        return detections

    def infer(self, frame_rgb):
        return [d.keypoints for d in self.infer_detections(frame_rgb)]

    def decode_keypoints(self, outputs):
        return [d.keypoints for d in decode_detections(outputs, self._input_shape[:2])]

    def describe_outputs(self):
        return "\n".join([f"input: {self.input_info.name} {self._input_shape}"]
                         + [f"output: {i.name} {tuple(i.shape)}" for i in self.output_infos])

    def close(self):
        self._resources.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
