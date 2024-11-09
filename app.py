from flask import Flask, render_template, Response
from yolov11_detector import YOLOv11CrowdDetector
import cv2
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Inisialisasi detektor dan kamera
try:
    detector = YOLOv11CrowdDetector()
except Exception as e:
    logging.error("Gagal menginisialisasi YOLOv11CrowdDetector: %s", e)
    detector = None

camera = cv2.VideoCapture(1)
if not camera.isOpened():
    logging.error("Kamera tidak dapat diakses")
    camera = None

def generate_frames():
    if not camera or not detector:
        logging.error("Streaming tidak dapat dimulai: Kamera atau detektor tidak diinisialisasi.")
        return

    while camera.isOpened():
        success, frame = camera.read()
        if not success:
            logging.warning("Gagal menangkap frame dari kamera.")
            break
        try:
            frame = detector.detect_and_annotate(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
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
