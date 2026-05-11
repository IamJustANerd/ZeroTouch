import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QRect, QTimer
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QFont, QLinearGradient, QPen
from hand_tracker import HandTrackerEngine
from audio_assist_ui import ChatAssistantWidget
import pyautogui

class TrackingThread(QThread):
    # Signal: frame, state, progress, mlp_gest, fps
    change_pixmap_signal = pyqtSignal(np.ndarray, str, float, str, float)

    def __init__(self, debug=True):
        super().__init__()
        self.debug = debug
        self.engine = HandTrackerEngine(debug=debug, callback=self.update_data)

    def update_data(self, img, state, progress, mlp_gest, fps):
        self.change_pixmap_signal.emit(img, state, progress, mlp_gest, fps)

    def run(self):
        self.engine.run()

    def stop(self):
        self.engine.running = False
        self.wait()

class ZeroTouchOverlay(QMainWindow):
    exit_requested = pyqtSignal()
    
    def __init__(self, debug=True, show_chat=False):
        super().__init__()
        self.debug = debug
        self.show_chat = show_chat
        self.initUI()
        
        self.thread = TrackingThread(debug=debug)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    def initUI(self):
        # Window Properties
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | 
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Dimensions (Matched to Launcher style)
        self.win_w = 400
        self.win_h = 800 if self.show_chat else 240
        self.resize(self.win_w, self.win_h)
        
        # Position at bottom-right
        screen = QApplication.primaryScreen().availableGeometry()
        margin_x, margin_y = 10, 10 
        self.move(screen.width() - self.win_w - margin_x, screen.height() - self.win_h - margin_y)

        # Central Layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # 1. Camera Feed Label
        self.label = QLabel("INITIALIZING...")
        self.label.setFixedSize(400, 300 if self.show_chat else 240)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("background-color: rgba(0,0,0,100); border-radius: 24px;")
        self.main_layout.addWidget(self.label)
        
        # 2. Chat UI (Optional)
        if self.show_chat:
            self.chat_widget = ChatAssistantWidget()
            self.main_layout.addWidget(self.chat_widget)
            # Link chat exit to main exit
            self.chat_widget.pill.exit_requested.connect(self.exit_requested.emit)

        # 3. Global Exit Button (Top Right of Camera)
        self.exit_btn = QPushButton("✕", self.label)
        self.exit_btn.setFixedSize(30, 30)
        self.exit_btn.move(self.win_w - 40, 10)
        self.exit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.exit_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 20);
                color: white;
                border-radius: 15px;
                font-size: 14px;
                border: none;
            }
            QPushButton:hover {
                background-color: rgba(255, 0, 0, 150);
            }
        """)
        self.exit_btn.clicked.connect(self.exit_requested.emit)

        # Periodic "Always on Top" reinforcement (1s interval)
        self.top_timer = QTimer(self)
        self.top_timer.timeout.connect(self.raise_)
        self.top_timer.start(1000)

        self.show()

    def update_image(self, cv_img, state, progress, mlp_gest, fps):
        try:
            # 1. Dimensions
            lbl_w = self.label.width()
            lbl_h = self.label.height()

            # 2. Conversion
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            display_img = cv2.resize(rgb_image, (lbl_w, lbl_h))
            h, w, ch = display_img.shape
            bytes_per_line = ch * w
            qt_img = QImage(display_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            
            # 3. Painter Logic
            canvas = QPixmap.fromImage(qt_img)
            painter = QPainter(canvas)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            # Info Bar Background
            bar_rect = QRect(10, h - 50, w - 20, 40)
            painter.setBrush(QColor(30, 30, 30, 180))
            painter.setPen(QPen(QColor(255, 255, 255, 40), 1))
            painter.drawRoundedRect(bar_rect, 20, 20)
            
            # Font
            font = QFont("Inter")
            if not font.exactMatch(): font = QFont("Segoe UI")
            font.setPointSize(10)
            font.setWeight(QFont.Weight.Bold)
            painter.setFont(font)
            
            # State & Gesture
            # Colors
            if state == "LISTENING":
                state_color = QColor(255, 255, 0) # Yellow
            elif state != "IDLE":
                state_color = QColor(0, 255, 255) # Cyan
            else:
                state_color = QColor(200, 200, 200) # Gray
                
            painter.setPen(state_color)
            painter.drawText(30, h - 23, f"STATE: {state}")
            
            painter.setPen(QColor(255, 255, 255, 180))
            painter.drawText(w - 140, h - 23, f"GEST: {mlp_gest}")
            painter.drawText(w - 220, h - 23, f"FPS: {int(fps)}")

            # 4. Circular Progress (Top-Left Corner of HUD)
            if progress > 0:
                ring_rect = QRect(20, 20, 40, 40)
                # Background
                painter.setPen(QPen(QColor(255, 255, 255, 40), 3))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(ring_rect)
                # Progress
                ring_color = QColor(255, 255, 0) if state == "LISTENING" else QColor(0, 255, 255)
                painter.setPen(QPen(ring_color, 3))
                span_angle = int(-progress * 360 * 16)
                painter.drawArc(ring_rect, 90 * 16, span_angle)
            
            painter.end()
            self.label.setPixmap(canvas)

            # 5. Update Chat Progress (If combined)
            if self.show_chat:
                self.chat_widget.set_progress(progress)

        except Exception as e:
            print(f"Render Error: {e}")

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ZeroTouchOverlay(debug=True)
    sys.exit(app.exec())
