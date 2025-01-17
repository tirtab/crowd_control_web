from concurrent.futures import ThreadPoolExecutor
from flask import Flask
from flask_mqtt import Mqtt
from crowd_detector import YOLOv11CrowdDetector
from fatigue_detector import YOLOv11FatigueDetector
from orjson import dumps, OPT_SERIALIZE_NUMPY
import orjson
import logging
import base64
from PIL import Image
import numpy as np
from io import BytesIO
import cv2

# Inisialisasi Flask dan Flask-MQTT
app = Flask(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Konfigurasi MQTT
app.config.update(
    {
        "MQTT_BROKER_URL": "localhost",
        "MQTT_BROKER_PORT": 1883,
        "MQTT_REFRESH_TIME": 1.0,
    }
)

mqtt = Mqtt(app)

# Subscription Topics
CROWD_FRAME_TOPIC = "mqtt-crowd-frame"
FATIGUE_FRAME_TOPIC = "mqtt-fatigue-frame"

# Publication Topics
CROWD_RESULT_TOPIC = "mqtt-crowd-result"
FATIGUE_RESULT_TOPIC = "mqtt-fatigue-result"


def initialize_crowd_detector():
    """Inisialisasi crowd detector."""
    try:
        logging.info("Menginisialisasi YOLOv11CrowdDetector...")
        return YOLOv11CrowdDetector()
    except Exception as e:
        logging.error(f"Gagal menginisialisasi YOLOv11CrowdDetector: {e}")
        return None


def initialize_fatigue_detector():
    """Inisialisasi fatigue detector."""
    try:
        logging.info("Menginisialisasi YOLOv11FatigueDetector...")
        return YOLOv11FatigueDetector()
    except Exception as e:
        logging.error(f"Gagal menginisialisasi YOLOv11FatigueDetector: {e}")
        return None


# Inisialisasi Detektor Secara Paralel
with ThreadPoolExecutor() as executor:
    future_crowd = executor.submit(initialize_crowd_detector)
    future_fatigue = executor.submit(initialize_fatigue_detector)
    crowd_detector = future_crowd.result()
    fatigue_detector = future_fatigue.result()

if crowd_detector and fatigue_detector:
    logging.info("Detektor berhasil diinisialisasi.")
else:
    logging.warning("Satu atau lebih detektor gagal diinisialisasi.")


# # Inisialisasi Detektor
# try:
#     crowd_detector = YOLOv11CrowdDetector()
#     fatigue_detector = YOLOv11FatigueDetector()
#     logging.info("Detektor berhasil diinisialisasi.")
# except Exception as e:
#     logging.error(f"Gagal menginisialisasi detektor: {e}")
#     crowd_detector, fatigue_detector = None, None


def process_frame(frame_data):
    """Decode base64 frame and convert to OpenCV format."""
    try:
        if "," in frame_data:
            frame_data = frame_data.split(",", 1)[1]
        frame_bytes = base64.b64decode(frame_data)
        frame_pil = Image.open(BytesIO(frame_bytes))
        return cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logging.error(f"Error processing frame: {e}")
        return None


@mqtt.on_connect()
def handle_connect(client, userdata, flags, rc):
    """Handle MQTT connection."""
    if rc == 0:
        logging.info("Connected to MQTT Broker.")
        mqtt.subscribe([(CROWD_FRAME_TOPIC, 0), (FATIGUE_FRAME_TOPIC, 0)])
        logging.info(
            f"Subscribed to topics: {CROWD_FRAME_TOPIC}, {FATIGUE_FRAME_TOPIC}"
        )
    else:
        logging.error(f"MQTT connection failed with code {rc}")


@mqtt.on_message()
def handle_mqtt_message(client, userdata, message):
    """Process MQTT messages."""
    topic = message.topic
    payload = message.payload.decode("utf-8")
    try:
        data = orjson.loads(payload)
        frame = process_frame(data)
        if frame is None:
            return

        if topic == CROWD_FRAME_TOPIC and crowd_detector:
            _, detections = crowd_detector.detect_and_annotate(frame)
            detections_list = [detection.tolist() if isinstance(detection, np.ndarray) else detection for detection in
                               detections]
            num_people = len(detections_list)

            mqtt_data = {"num_people": num_people, "detection": detections_list}
            mqtt.publish(
                CROWD_RESULT_TOPIC, dumps(mqtt_data, option=OPT_SERIALIZE_NUMPY)
            )

        elif topic == FATIGUE_FRAME_TOPIC and fatigue_detector:
            detections = fatigue_detector.detect_and_annotate(frame)
            fatigue_status = fatigue_detector.get_fatigue_category(detections)
            detections_list = [detection.tolist() if isinstance(detection, np.ndarray) else detection for detection in
                               detections]

            mqtt_data = {"status": fatigue_status, "detection": detections_list}
            mqtt.publish(
                FATIGUE_RESULT_TOPIC, dumps(mqtt_data, option=OPT_SERIALIZE_NUMPY)
            )

    except Exception as e:
        logging.error(f"Error processing MQTT message: {e}")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
