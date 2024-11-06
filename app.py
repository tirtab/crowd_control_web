from flask import Flask, render_template, Response
from yolov11_detector import YOLOv11CrowdDetector
import cv2

app = Flask(__name__)

# Inisialisasi detektor YOLOv11
detector = YOLOv11CrowdDetector()

# Baca input video dari webcam
camera = cv2.VideoCapture(1)  # Bisa diubah ke ID kamera atau URL stream


def generate_frames():
    while True:
        # Baca frame dari kamera
        success, frame = camera.read()
        if not success:
            break

        # Jalankan deteksi menggunakan YOLOv11
        frame = detector.detect_and_annotate(frame)

        # Encode frame ke format JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Streaming frame sebagai video stream MJPEG
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    # Halaman utama untuk menampilkan video stream
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    # Streaming video
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
