import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from pathlib import Path
import openvino as ov
import logging

ZONE_POLYGON = np.array([
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1]
])


class YOLOv11CrowdDetector:
    def __init__(self):
        self.frame_width = 640
        self.frame_height = 480

        # Inisialisasi model OpenVINO
        try:
            model_path = Path("model/best_openvino_model(Yolov11m)/best.xml")
            core = ov.Core()
            model = core.read_model(model_path)

            self.device = "AUTO"
            ov_config = {}

            # Konfigurasi perangkat
            if self.device != "CPU":
                model.reshape({0: [1, 3, 640, 640]})
            if "GPU" in self.device or ("AUTO" in self.device and "GPU" in core.available_devices):
                ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}

            compiled_model = core.compile_model(model, self.device, ov_config)
            self.det_model = YOLO(model_path.parent, task="detect")
            self.det_model.predictor.model.ov_compiled_model = compiled_model
        except Exception as e:
            logging.error("Error saat memuat model OpenVINO: %s", e)
            raise

        # Inisialisasi anotator
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator()

        zone_polygon = (ZONE_POLYGON * np.array([self.frame_width, self.frame_height])).astype(int)
        self.zone = sv.PolygonZone(polygon=zone_polygon)
        self.zone_annotator = sv.PolygonZoneAnnotator(
            zone=self.zone,
            color=sv.Color.RED,
            thickness=2,
            text_thickness=4,
            text_scale=2,
        )

    def detect_and_annotate(self, frame):
        try:
            result = self.det_model(frame)[0]
            detections = sv.Detections.from_ultralytics(result).with_nms().with_nmm()
            detections = detections[detections.confidence > 0.5]

            # Anotasi bounding box dan label
            labels = [
                f"{class_name} {confidence: .2f}"
                for class_name, confidence
                in zip(detections['class_name'], detections.confidence)
            ]

            frame = self.box_annotator.annotate(scene=frame, detections=detections)
            frame = self.label_annotator.annotate(scene=frame, detections=detections, labels=labels)

            # Anotasi zona deteksi
            self.zone.trigger(detections=detections)
            frame = self.zone_annotator.annotate(scene=frame)

            return frame
        except Exception as e:
            logging.error("Error saat deteksi dan anotasi: %s", e)
            return frame  # Kembalikan frame asli jika terjadi error
