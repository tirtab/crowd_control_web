from flask import Flask, render_template, Response
from crowd_detector import YOLOv11CrowdDetector
from fatigue_detector import YOLOv11FatigueDetector
import cv2
import logging
import paho.mqtt.client as mqtt
import json
from datetime import datetime
import base64
from PIL import Image
import numpy as np
from io import BytesIO

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Inisialisasi detektor dan kamera
try:
    crowd_detector = YOLOv11CrowdDetector()
    fatigue_detector = YOLOv11FatigueDetector()
except Exception as e:
    logging.error("Gagal menginisialisasi detektor: %s", e)
    crowd_detector = None
    fatigue_detector = None

# MQTT setup
mqtt_client = mqtt.Client()
mqtt_client.connect("localhost", 1883, 60)

camera = cv2.VideoCapture(1)
if not camera.isOpened():
    logging.error("Kamera tidak dapat diakses")
    camera = None

# Helper untuk validasi Base64
# def is_valid_base64(base64_string):
#     try:
#         # Tambahkan padding jika panjang string tidak kelipatan 4
#         if len(base64_string) % 4 != 0:
#             base64_string += '=' * (4 - len(base64_string) % 4)
#         base64.b64decode(base64_string)  # Coba decode
#         return True
#     except Exception as e:
#         logging.error(f"Base64 decoding error: {e}")
#         return False


# Fungsi untuk menangani frame video
def process_frame(frame_data):
    try:
        # Hapus awalan data URI (jika ada)
        if ',' in frame_data:
            frame_data = frame_data.split(',')[1]

        # # Validasi Base64 string
        # if not is_valid_base64(frame_data):
        #     logging.error("Invalid Base64 string received, skipping frame processing.")
        #     return None

        # Decode Base64 menjadi frame
        frame_bytes = base64.b64decode(frame_data)
        frame_pil = Image.open(BytesIO(frame_bytes))
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        return frame
    except Exception as e:
        logging.error(f"Error processing frame: {e}")
        return None


# Fungsi untuk menghasilkan streaming crowd analysis
def generate_crowd_frames():
    if not camera or not crowd_detector:
        logging.error("Streaming tidak dapat dimulai: Kamera atau detektor tidak diinisialisasi.")
        return

    while camera.isOpened():
        success, frame = camera.read()
        if not success:
            logging.warning("Gagal menangkap frame dari kamera.")
            break
        try:
            # Deteksi dan anotasi frame
            frame, detection_data = crowd_detector.detect_and_annotate(frame)
            num_people = len(detection_data)

            # Publikasikan hasil analisis ke MQTT
            mqtt_data = {
                "status": "success",
                "timestamp": str(datetime.now()),
                "num_people": num_people,
                "detections": detection_data
            }
            mqtt_client.publish("video/analysis", json.dumps(mqtt_data))

            # Encode frame untuk streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as e:
            logging.error("Error dalam memproses frame: %s", e)
            break


# Fungsi untuk menghasilkan streaming fatigue analysis
def generate_fatigue_frames():
    if not camera or not fatigue_detector:
        logging.error("Streaming tidak dapat dimulai: Kamera atau detektor tidak diinisialisasi.")
        return

    while camera.isOpened():
        success, frame = camera.read()
        if not success:
            logging.warning("Gagal menangkap frame dari kamera.")
            break
        try:
            # Deteksi kelelahan
            frame, detected_classes = fatigue_detector.detect_and_annotate(frame)
            fatigue_status = fatigue_detector.get_fatigue_category(detected_classes)

            # Publikasikan hasil ke MQTT
            mqtt_data = {
                "status": fatigue_status,
                "timestamp": str(datetime.now())
            }
            mqtt_client.publish("fatigue-analysis", json.dumps(mqtt_data))

            # Tambahkan status ke frame
            cv2.putText(frame, fatigue_status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Encode frame untuk streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as e:
            logging.error("Error dalam memproses frame: %s", e)
            break


# Callback untuk menangani pesan MQTT
def on_message(client, userdata, message):
    try:
        logging.info(f"Received message on topic {message.topic}")
        frame_data = message.payload.decode()

        # Proses frame menggunakan Base64 decoder
        frame = process_frame(frame_data)
        if frame is None:
            return

        # Deteksi keramaian
        frame, detection_data = crowd_detector.detect_and_annotate(frame)
        num_people = len(detection_data)

        # Publikasikan hasil analisis ke MQTT
        mqtt_data = {
            "status": "success",
            "timestamp": str(datetime.now()),
            "num_people": num_people,
            "detections": detection_data
        }
        mqtt_client.publish("video/analysis", json.dumps(mqtt_data))
    except Exception as e:
        logging.error(f"Error in MQTT message handling: {e}")
        mqtt_client.publish("video/analysis", json.dumps({
            "status": "error",
            "timestamp": str(datetime.now()),
            "error": str(e)
        }))


# Set up MQTT subscriber
mqtt_client.on_message = on_message
mqtt_client.subscribe('video/frames')
mqtt_client.loop_start()

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/crowd_analysis')
def crowd_analysis():
    return render_template('crowd_analysis.html')


@app.route('/fatigue_analysis')
def fatigue_analysis():
    return render_template('fatigue_analysis.html')


@app.route('/video_feed/crowd')
def video_feed_crowd():
    return Response(generate_crowd_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed/fatigue')
def video_feed_fatigue():
    return Response(generate_fatigue_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True, port=5000)
