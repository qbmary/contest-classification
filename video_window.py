import sys
import time
from typing import Optional
from pathlib import Path

import cv2
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
SNAPSHOTS_DIR = OUTPUTS_DIR / "snapshots"


class InfoCard(QFrame):
    def __init__(self, title: str, value: str):
        super().__init__()
        self.setObjectName("InfoCard")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(6)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("CardTitle")

        self.value_label = QLabel(value)
        self.value_label.setObjectName("CardValue")
        self.value_label.setWordWrap(True)

        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)

    def set_value(self, value: str):
        self.value_label.setText(value)


class VideoLabel(QLabel):
    DEFAULT_TEXT = (
        "Видеопоток с камеры\n\n"
        "После нажатия кнопки «Запуск распознавания»\n"
        "здесь появится изображение с веб-камеры"
    )

    def __init__(self):
        super().__init__()
        self.setObjectName("VideoFrame")
        self.setMinimumSize(920, 540)
        self.setAlignment(Qt.AlignCenter)
        self.setWordWrap(True)
        self.setText(self.DEFAULT_TEXT)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int, img_size: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        feature_size = img_size // 8

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * feature_size * feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Vision — Распознавание в видеопотоке")
        self.resize(1470, 900)
        self.setMinimumSize(1290, 780)

        self.cap: Optional[cv2.VideoCapture] = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.last_time = 0.0
        self.current_camera_index = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.class_names = []
        self.img_size = 224
        self.model_loaded = False

        self.last_prediction = "—"
        self.last_confidence = 0.0
        self.frame_transform = None

        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

        root = QWidget()
        self.setCentralWidget(root)

        page = QHBoxLayout(root)
        page.setContentsMargins(22, 22, 22, 22)
        page.setSpacing(18)

        self.left_panel = self.build_left_panel()
        self.right_panel = self.build_right_panel()

        page.addWidget(self.left_panel, 4)
        page.addWidget(self.right_panel, 1)

        self.apply_styles()
        self.populate_cameras()
        self.reset_video_placeholder()

    def build_left_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        header = QFrame()
        header.setObjectName("HeaderPanel")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 18, 20, 18)

        title_box = QVBoxLayout()
        title_box.setSpacing(4)

        title = QLabel("Распознавание в видеопотоке")
        title.setObjectName("MainTitle")

        subtitle = QLabel(
            "Главное окно приложения: камера, запуск модели, вывод FPS и уверенности"
        )
        subtitle.setObjectName("Subtitle")

        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header_layout.addLayout(title_box)
        header_layout.addStretch()

        self.status_chip = QLabel("Статус: готово")
        self.status_chip.setObjectName("StatusChip")
        self.status_chip.setAlignment(Qt.AlignCenter)
        self.status_chip.setMinimumWidth(150)
        header_layout.addWidget(self.status_chip)

        self.video_label = VideoLabel()

        metrics = QWidget()
        metrics_layout = QGridLayout(metrics)
        metrics_layout.setContentsMargins(0, 0, 0, 0)
        metrics_layout.setHorizontalSpacing(14)
        metrics_layout.setVerticalSpacing(14)

        self.fps_card = InfoCard("FPS", "0.0")
        self.conf_card = InfoCard("Уверенность", "—")
        self.model_card = InfoCard("Модель", "Не выбрана")
        self.result_card = InfoCard("Результат", "Ожидание запуска")

        metrics_layout.addWidget(self.fps_card, 0, 0)
        metrics_layout.addWidget(self.conf_card, 0, 1)
        metrics_layout.addWidget(self.model_card, 1, 0)
        metrics_layout.addWidget(self.result_card, 1, 1)

        bottom_note = QFrame()
        bottom_note.setObjectName("HintPanel")
        bottom_layout = QVBoxLayout(bottom_note)
        bottom_layout.setContentsMargins(18, 16, 18, 16)

        hint_title = QLabel("Дополнительная информация")
        hint_title.setObjectName("SectionTitle")

        self.hint_text = QLabel(
            "Этот блок можно использовать для логов, подсказок пользователю, "
            "ошибок загрузки модели и параметров камеры."
        )
        self.hint_text.setObjectName("HintText")
        self.hint_text.setWordWrap(True)

        bottom_layout.addWidget(hint_title)
        bottom_layout.addWidget(self.hint_text)

        layout.addWidget(header)
        layout.addWidget(self.video_label, 1)
        layout.addWidget(metrics)
        layout.addWidget(bottom_note)
        return container

    def make_field_block(self, label: QLabel, widget: QWidget) -> QWidget:
        block = QWidget()
        layout = QVBoxLayout(block)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(6)
        layout.addWidget(label)
        layout.addWidget(widget)
        return block

    def build_right_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("SidePanel")
        panel.setMinimumWidth(380)
        panel.setMaximumWidth(460)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        camera_group = QGroupBox("Настройки камеры")
        camera_group.setObjectName("SettingsGroup")
        camera_layout = QVBoxLayout(camera_group)
        camera_layout.setContentsMargins(12, 14, 12, 12)
        camera_layout.setSpacing(10)

        cam_label = QLabel("Источник видео")
        cam_label.setObjectName("FieldLabel")
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumHeight(46)

        resolution_label = QLabel("Разрешение")
        resolution_label.setObjectName("FieldLabel")
        self.resolution_combo = QComboBox()
        self.resolution_combo.setMinimumHeight(46)
        self.resolution_combo.addItems(["640 × 480", "1280 × 720", "1920 × 1080"])

        brightness_label = QLabel("Яркость")
        brightness_label.setObjectName("FieldLabel")
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(0, 100)
        self.brightness_slider.setValue(55)
        self.brightness_slider.setFixedHeight(22)

        self.show_overlay = QCheckBox("Показывать наложение информации")
        self.show_overlay.setChecked(True)

        camera_layout.addWidget(self.make_field_block(cam_label, self.camera_combo))
        camera_layout.addWidget(self.make_field_block(resolution_label, self.resolution_combo))
        camera_layout.addWidget(self.make_field_block(brightness_label, self.brightness_slider))
        camera_layout.addWidget(self.show_overlay)

        model_group = QGroupBox("Модель нейронной сети")
        model_group.setObjectName("SettingsGroup")
        model_layout = QVBoxLayout(model_group)
        model_layout.setContentsMargins(12, 14, 12, 12)
        model_layout.setSpacing(10)

        model_label = QLabel("Выбор модели")
        model_label.setObjectName("FieldLabel")
        self.model_combo = QComboBox()
        self.model_combo.setMinimumHeight(46)
        self.model_combo.addItems([
            "classifier_finetune.pth",
            "classifier_scratch.pth",
        ])

        model_layout.addWidget(self.make_field_block(model_label, self.model_combo))

        control_group = QGroupBox("Управление")
        control_group.setObjectName("SettingsGroup")
        control_layout = QVBoxLayout(control_group)
        control_layout.setContentsMargins(12, 14, 12, 12)
        control_layout.setSpacing(12)

        self.start_btn = QPushButton("Запуск распознавания")
        self.stop_btn = QPushButton("Остановить")
        self.stop_btn.setObjectName("SecondaryButton")
        self.snapshot_btn = QPushButton("Сделать снимок кадра")
        self.snapshot_btn.setObjectName("SecondaryButton")

        self.start_btn.clicked.connect(self.start_camera)
        self.stop_btn.clicked.connect(self.stop_camera)
        self.snapshot_btn.clicked.connect(self.save_snapshot)

        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.snapshot_btn)

        app_group = QGroupBox("Служебная информация")
        app_group.setObjectName("SettingsGroup")
        app_layout = QVBoxLayout(app_group)
        app_layout.setContentsMargins(12, 14, 12, 12)
        app_layout.setSpacing(8)

        app_text = QLabel(
            "Окно соответствует требованиям конкурса: выбор камеры, загрузка модели, "
            "запуск распознавания, отображение видеопотока, FPS и уверенности."
        )
        app_text.setObjectName("HintText")
        app_text.setWordWrap(True)
        app_layout.addWidget(app_text)

        layout.addWidget(camera_group)
        layout.addWidget(model_group)
        layout.addWidget(control_group)
        layout.addWidget(app_group)
        layout.addStretch()
        return panel

    def reset_video_placeholder(self):
        self.video_label.clear()
        self.video_label.setPixmap(QPixmap())
        self.video_label.setText(VideoLabel.DEFAULT_TEXT)

    def populate_cameras(self):
        self.camera_combo.clear()
        found_any = False

        for index in range(4):
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if cap.isOpened():
                self.camera_combo.addItem(f"Камера {index}", index)
                found_any = True
            cap.release()

        if not found_any:
            self.camera_combo.addItem("Камера 0", 0)

    def parse_resolution(self):
        text = self.resolution_combo.currentText().replace(" ", "")
        width, height = text.split("×")
        return int(width), int(height)

    def build_transform(self):
        self.frame_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])

    def load_selected_model(self):
        model_name = self.model_combo.currentText()
        model_path = MODELS_DIR / model_name

        if not model_path.exists():
            QMessageBox.warning(self, "Ошибка", f"Файл модели не найден:\n{model_path}")
            self.status_chip.setText("Статус: ошибка")
            self.result_card.set_value("Модель не загружена")
            self.model = None
            self.class_names = []
            self.model_loaded = False
            return False

        checkpoint = torch.load(model_path, map_location=self.device)
        self.class_names = checkpoint["class_names"]
        self.img_size = checkpoint["img_size"]

        if model_name == "classifier_finetune.pth":
            model = models.resnet18(weights=None)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, len(self.class_names))

        elif model_name == "classifier_scratch.pth":
            model = SimpleCNN(num_classes=len(self.class_names), img_size=self.img_size)

        else:
            QMessageBox.warning(self, "Ошибка", "Неизвестный тип модели.")
            self.status_chip.setText("Статус: ошибка")
            self.result_card.set_value("Модель не загружена")
            self.model = None
            self.class_names = []
            self.model_loaded = False
            return False

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        self.model = model
        self.model_loaded = True
        self.build_transform()

        self.hint_text.setText(
            f"Загружена модель: {model_name}. "
            f"Классы: {', '.join(self.class_names)}"
        )
        return True

    def predict_frame(self, frame):
        if self.model is None or self.frame_transform is None:
            return "—", 0.0

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image_tensor = self.frame_transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)

        predicted_class = self.class_names[predicted_idx.item()]
        confidence_value = confidence.item()
        return predicted_class, confidence_value

    def start_camera(self):
        self.stop_camera()

        camera_index = self.camera_combo.currentData()
        if camera_index is None:
            camera_index = 0
        self.current_camera_index = int(camera_index)

        self.cap = cv2.VideoCapture(self.current_camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.current_camera_index)

        if not self.cap.isOpened():
            QMessageBox.critical(self, "Ошибка", "Не удалось открыть выбранную камеру.")
            self.status_chip.setText("Статус: ошибка")
            self.result_card.set_value("Камера не запущена")
            return

        width, height = self.parse_resolution()
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.load_selected_model():
            self.cap.release()
            self.cap = None
            return

        self.last_time = time.time()
        self.timer.start(30)

        self.status_chip.setText("Статус: камера активна")
        self.model_card.set_value(self.model_combo.currentText())
        self.result_card.set_value("Видеопоток запущен")
        self.hint_text.setText(
            f"Камера {self.current_camera_index} запущена. "
            f"Разрешение: {width}×{height}. "
            f"Модель: {self.model_combo.currentText()}. "
            f"Распознавание активно."
        )

    def stop_camera(self):
        self.timer.stop()

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.reset_video_placeholder()

        self.fps_card.set_value("0.0")
        self.conf_card.set_value("—")
        self.model_card.set_value("Не выбрана")
        self.result_card.set_value("Ожидание запуска")

        self.last_prediction = "—"
        self.last_confidence = 0.0

        self.model = None
        self.class_names = []
        self.frame_transform = None
        self.model_loaded = False

        self.status_chip.setText("Статус: готово")

    def save_snapshot(self):
        if self.cap is None:
            QMessageBox.information(self, "Информация", "Сначала запустите камеру.")
            return

        ok, frame = self.cap.read()
        if not ok:
            QMessageBox.warning(self, "Ошибка", "Не удалось получить кадр для сохранения.")
            return

        filename = SNAPSHOTS_DIR / f"snapshot_{int(time.time())}.jpg"
        cv2.imwrite(str(filename), frame)
        self.hint_text.setText(f"Снимок сохранён в файл: {filename}")

    def update_frame(self):
        if self.cap is None:
            return

        ok, frame = self.cap.read()
        if not ok:
            self.stop_camera()
            QMessageBox.warning(self, "Ошибка", "Поток с камеры был прерван.")
            return

        frame = self.apply_brightness(frame)

        now = time.time()
        dt = max(now - self.last_time, 1e-6)
        fps = 1.0 / dt
        self.last_time = now

        predicted_class, confidence = self.predict_frame(frame)
        self.last_prediction = predicted_class
        self.last_confidence = confidence

        self.fps_card.set_value(f"{fps:.1f}")
        self.conf_card.set_value(f"{confidence:.2f}")
        self.result_card.set_value(predicted_class)

        display_frame = frame.copy()
        if self.show_overlay.isChecked():
            self.draw_overlay(display_frame, fps)

        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        scaled = qt_image.scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        self.video_label.setPixmap(QPixmap.fromImage(scaled))
        self.video_label.setText("")

    def apply_brightness(self, frame):
        slider_value = self.brightness_slider.value()
        beta = int((slider_value - 50) * 2.2)
        return cv2.convertScaleAbs(frame, alpha=1.0, beta=beta)

    def draw_overlay(self, frame, fps: float):
        confidence = self.last_confidence
        result = self.last_prediction

        cv2.rectangle(frame, (20, 20), (460, 125), (15, 23, 42), thickness=-1)

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (35, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Confidence: {confidence:.2f}",
            (35, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Result: {result}",
            (35, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()

    def apply_styles(self):
        self.setStyleSheet(
            """
            QWidget {
                background: #0f172a;
                color: #e5e7eb;
                font-family: 'Segoe UI';
                font-size: 14px;
            }
            QFrame#HeaderPanel, QFrame#SidePanel, QFrame#HintPanel, QFrame#InfoCard {
                background: #111827;
                border: 1px solid #1f2937;
                border-radius: 18px;
            }
            QLabel#MainTitle {
                font-size: 28px;
                font-weight: 700;
                color: #f8fafc;
            }
            QLabel#Subtitle {
                color: #94a3b8;
                font-size: 13px;
            }
            QLabel#StatusChip {
                background: #052e16;
                color: #86efac;
                border: 1px solid #14532d;
                border-radius: 12px;
                padding: 10px 14px;
                font-weight: 600;
            }
            QLabel#VideoFrame {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #111827,
                    stop:1 #1e293b
                );
                border: 2px dashed #334155;
                border-radius: 22px;
                color: #94a3b8;
                font-size: 18px;
                line-height: 1.5;
                padding: 14px;
            }
            QLabel#CardTitle, QLabel#SectionTitle, QLabel#FieldLabel {
                color: #94a3b8;
                font-size: 12px;
                font-weight: 600;
                text-transform: uppercase;
            }
            QLabel#CardValue {
                color: #f8fafc;
                font-size: 22px;
                font-weight: 700;
            }
            QLabel#HintText {
                color: #cbd5e1;
                font-size: 13px;
            }
            QGroupBox#SettingsGroup {
                background: #0b1220;
                border: 1px solid #1f2937;
                border-radius: 16px;
                margin-top: 14px;
                padding-top: 16px;
                font-weight: 700;
                color: #f8fafc;
            }
            QGroupBox#SettingsGroup::title {
                subcontrol-origin: margin;
                left: 14px;
                top: -2px;
                padding: 0 6px;
            }
            QComboBox, QPushButton {
                min-height: 44px;
                border-radius: 12px;
            }
            QComboBox {
                background: #111827;
                border: 1px solid #334155;
                padding: 8px 34px 8px 12px;
            }
            QComboBox QAbstractItemView {
                background: #111827;
                border: 1px solid #334155;
                selection-background-color: #1d4ed8;
                padding: 4px;
            }
            QComboBox::drop-down {
                border: none;
                width: 28px;
            }
            QPushButton {
                background: #2563eb;
                border: none;
                color: white;
                font-weight: 700;
                padding: 0 14px;
            }
            QPushButton:hover {
                background: #1d4ed8;
            }
            QPushButton#SecondaryButton {
                background: #1f2937;
                border: 1px solid #334155;
                color: #e5e7eb;
            }
            QPushButton#SecondaryButton:hover {
                background: #273449;
            }
            QCheckBox {
                spacing: 10px;
                color: #cbd5e1;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QSlider {
                background: transparent;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #1f2937;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #60a5fa;
                width: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }
            """
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec_())