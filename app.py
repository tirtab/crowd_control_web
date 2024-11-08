from flask import Flask, render_template, Response
from yolov11_detector import YOLOv11CrowdDetector
import cv2
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Inisialisasi detektor YOLOv11 dengan penanganan error
try:
    detector = YOLOv11CrowdDetector()
except Exception as e:
    logging.error("Gagal menginisialisasi YOLOv11CrowdDetector: %s", e)
    detector = None

# Konfigurasi sumber kamera, gunakan default jika tidak tersedia
CAMERA_ID = 1
try:
    camera = cv2.VideoCapture(CAMERA_ID)
    if not camera.isOpened():
        raise ValueError("Kamera tidak dapat diakses")
except Exception as e:
    logging.error("Tidak dapat membuka kamera %s: %s", CAMERA_ID, e)
    camera = None

def generate_frames():
    if camera is None or detector is None:
        logging.error("Streaming tidak dapat dimulai: Kamera atau detektor tidak diinisialisasi.")
        return b""

    while True:
        success, frame = camera.read()
        if not success:
            logging.warning("Gagal menangkap frame dari kamera.")
            break

        try:
            frame = detector.detect_and_annotate(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                logging.warning("Gagal mengenkode frame ke format JPEG.")
                continue

            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            logging.error("Error dalam memproses frame: %s", e)
            break

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
