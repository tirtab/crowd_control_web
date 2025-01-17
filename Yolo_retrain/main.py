import os
import smtplib
import ssl
import schedule
import time
import logging
from datetime import datetime
import mlflow
import mlflow.pytorch
from ultralytics import YOLO
import torch
import subprocess
import platform


class AdvancedYOLOTrainer:
    def __init__(self,
                 model_path='yolov11.pt',
                 dataset_path='dataset.yaml',
                 experiment_name='yolo_automated_training',
                 email_config=None):
        """
        Inisialisasi trainer dengan konfigurasi email

        Args:
            model_path (str): Path model YOLO awal
            dataset_path (str): Path konfigurasi dataset
            experiment_name (str): Nama eksperimen MLflow
            email_config (dict): Konfigurasi email untuk notifikasi
        """
        # Konfigurasi logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )

        # Konfigurasi MLflow
        mlflow.set_tracking_uri("file:/./mlruns")
        mlflow.set_experiment(experiment_name)

        # Variabel konfigurasi
        self.model_path = model_path
        self.dataset_path = dataset_path

        # Direktori untuk menyimpan model
        self.model_save_dir = 'trained_models'
        os.makedirs(self.model_save_dir, exist_ok=True)

        # Konfigurasi email default
        self.email_config = email_config or {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'sender_email': 'your_email@gmail.com',
            'sender_password': 'your_app_password',
            'recipient_email': 'recipient@example.com'
        }

    def send_email_notification(self, subject, body):
        """
        Kirim notifikasi email
        """
        try:
            context = ssl.create_default_context()
            with smtplib.SMTP(self.email_config['smtp_server'],
                              self.email_config['smtp_port']) as server:
                server.starttls(context=context)
                server.login(self.email_config['sender_email'],
                             self.email_config['sender_password'])

                full_message = f"Subject: {subject}\n\n{body}"
                server.sendmail(
                    self.email_config['sender_email'],
                    self.email_config['recipient_email'],
                    full_message
                )
            logging.info("Email notifikasi terkirim")
        except Exception as e:
            logging.error(f"Gagal mengirim email: {e}")

    def train_model(self):
        """
        Proses training model dengan MLflow tracking dan email alert
        """
        try:
            # Mulai run MLflow
            with mlflow.start_run():
                start_time = datetime.now()
                logging.info(f"Memulai training model pada {start_time}")

                # Kirim email notifikasi awal
                self.send_email_notification(
                    "YOLO Training Dimulai",
                    f"Training model dimulai pada {start_time}"
                )

                # Load model
                model = YOLO(self.model_path)

                # Parameter training
                training_params = {
                    'data': self.dataset_path,
                    'epochs': 50,
                    'batch': 16,
                    'imgsz': 640,
                    'patience': 10,
                    'save_period': 5
                }

                # Log parameter training
                mlflow.log_params({
                    "model_path": self.model_path,
                    "dataset": self.dataset_path,
                    "epochs": training_params['epochs'],
                    "batch_size": training_params['batch'],
                    "image_size": training_params['imgsz']
                })

                # Jalankan training
                results = model.train(**training_params)

                # Hitung durasi training
                end_time = datetime.now()
                training_duration = end_time - start_time

                # Log metrics
                if hasattr(results, 'results_dict'):
                    metrics = results.results_dict
                    mlflow.log_metrics({
                        "mAP50": metrics.get('metrics/mAP50', 0),
                        "mAP50-95": metrics.get('metrics/mAP50-95', 0),
                        "precision": metrics.get('metrics/precision', 0),
                        "recall": metrics.get('metrics/recall', 0)
                    })

                # Simpan model
                model_save_path = os.path.join(
                    self.model_save_dir,
                    f'yolo_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt'
                )
                torch.save(model.model.state_dict(), model_save_path)

                # Log artifact model
                mlflow.log_artifact(model_save_path, "trained_models")

                # Kirim email notifikasi selesai
                self.send_email_notification(
                    "YOLO Training Selesai",
                    f"""
Training model selesai:
- Waktu Mulai: {start_time}
- Waktu Selesai: {end_time}
- Durasi: {training_duration}
- Metrics:
  * mAP50: {metrics.get('metrics/mAP50', 0)}
  * mAP50-95: {metrics.get('metrics/mAP50-95', 0)}
  * Precision: {metrics.get('metrics/precision', 0)}
  * Recall: {metrics.get('metrics/recall', 0)}
Model tersimpan di: {model_save_path}
                    """
                )

                logging.info("Training selesai dengan sukses")
                return results

        except Exception as e:
            error_msg = f"Kesalahan selama training: {e}"
            logging.error(error_msg)

            # Kirim email error
            self.send_email_notification(
                "YOLO Training Error",
                error_msg
            )
            return None

    def create_systemd_service(self):
        """
        Membuat systemd service untuk menjalankan training otomatis
        (Untuk Linux)
        """
        service_content = f"""[Unit]
Description=YOLO Automated Training Service
After=network.target

[Service]
ExecStart=/usr/bin/python3 {os.path.abspath(__file__)}
Restart=always
User={os.getlogin()}
WorkingDirectory={os.getcwd()}

[Install]
WantedBy=multi-user.target
"""

        service_path = "/etc/systemd/system/yolo-training.service"

        try:
            with open(service_path, 'w') as f:
                f.write(service_content)

            # Reload systemd, enable dan start service
            subprocess.run(["sudo", "systemctl", "daemon-reload"])
            subprocess.run(["sudo", "systemctl", "enable", "yolo-training.service"])
            subprocess.run(["sudo", "systemctl", "start", "yolo-training.service"])

            print("Systemd service berhasil dibuat dan dijalankan")
        except Exception as e:
            print(f"Gagal membuat systemd service: {e}")

    def schedule_training(self,
                          daily_time='02:00',
                          weekly_day='Monday',
                          monthly_date=1):
        """
        Jadwalkan training dengan berbagai opsi
        """
        # Training Harian
        schedule.every().day.at(daily_time).do(self.train_model)

        # Training Mingguan
        if weekly_day.lower() in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:
            getattr(schedule.every(), weekly_day.lower()).at(daily_time).do(self.train_model)

        # Training Bulanan
        def monthly_training():
            if datetime.now().day == monthly_date:
                self.train_model()

        schedule.every().day.do(monthly_training)

    def run_scheduler(self):
        """
        Jalankan scheduler training dengan manajemen sistem
        """
        logging.info("Memulai scheduler training YOLO")

        # Cek sistem operasi untuk manajemen proses
        os_name = platform.system()

        if os_name == 'Linux':
            # Untuk Linux, gunakan systemd
            self.create_systemd_service()
        elif os_name == 'Windows':
            # Untuk Windows, gunakan Windows Task Scheduler
            self._create_windows_scheduler()

        # Jalankan scheduler
        while True:
            schedule.run_pending()
            time.sleep(1)

    def _create_windows_scheduler(self):
        """
        Membuat tugas terjadwal di Windows
        """
        try:
            script_path = os.path.abspath(__file__)
            subprocess.run([
                'schtasks', '/create',
                '/tn', 'YOLOAutomatedTraining',
                '/tr', f'python "{script_path}"',
                '/sc', 'daily',
                '/st', '02:00'
            ])
            print("Windows scheduler berhasil dibuat")
        except Exception as e:
            print(f"Gagal membuat Windows scheduler: {e}")


# Contoh Penggunaan
if __name__ == '__main__':
    trainer = AdvancedYOLOTrainer(
        model_path='yolov11.pt',
        dataset_path='custom_dataset.yaml',
        experiment_name='yolo_automated_training',
        email_config={
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'sender_email': 'your_email@gmail.com',
            'sender_password': 'your_app_password',
            'recipient_email': 'recipient@example.com'
        }
    )

    # Jadwalkan training
    trainer.schedule_training(
        daily_time='02:00',  # Training jam 2 pagi
        weekly_day='Monday',  # Setiap hari Senin
        monthly_date=1  # Tanggal 1 setiap bulan
    )

    # Jalankan scheduler dengan manajemen sistem
    trainer.run_scheduler()