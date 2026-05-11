import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QFrame, QScrollArea, QLineEdit, QPushButton)
from PyQt6.QtCore import Qt, QTimer, QRect, QPoint, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QFont, QLinearGradient

class ChatBubble(QFrame):
    def __init__(self, text, is_user=True, parent=None):
        super().__init__(parent)
        self.is_user = is_user
        self.initUI(text)

    def initUI(self, text):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        
        self.label = QLabel(text)
        self.label.setWordWrap(True)
        self.label.setFixedWidth(280)
        
        if self.is_user:
            self.label.setStyleSheet("""
                background-color: rgba(0, 255, 255, 30);
                color: white;
                border: 1px solid rgba(0, 255, 255, 60);
                border-radius: 15px;
                padding: 10px;
                font-family: 'Inter';
                font-size: 13px;
            """)
            layout.addStretch()
            layout.addWidget(self.label)
        else:
            self.label.setStyleSheet("""
                background-color: rgba(255, 255, 255, 15);
                color: rgba(255, 255, 255, 220);
                border: 1px solid rgba(255, 255, 255, 30);
                border-radius: 15px;
                padding: 10px;
                font-family: 'Inter';
                font-size: 13px;
            """)
            layout.addWidget(self.label)
            layout.addStretch()

class VoicePill(QFrame):
    message_sent = pyqtSignal(str)
    exit_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pulse_val = 0
        self.status = "STATIC" # STATIC, LISTENING, THINKING, SPEAKING
        self.initUI()
        
        self.pulse_timer = QTimer(self)
        self.pulse_timer.timeout.connect(self.update_pulse)
        self.pulse_timer.start(50)

    def initUI(self):
        self.setFixedHeight(60)
        self.setFixedWidth(360)
        self.is_recording = False
        self.progress = 0.0 # 0.0 to 1.0
        
        self.setStyleSheet("background: transparent; border: none;")
        
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(15, 0, 10, 0)
        self.layout.setSpacing(10)

        # Pulse Indicator
        self.indicator = QLabel()
        self.indicator.setFixedSize(24, 24)
        self.indicator.setStyleSheet("background: transparent; border: none;")
        self.layout.addWidget(self.indicator)

        # Interactive Input
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type or speak...")
        self.input_field.setStyleSheet("""
            QLineEdit {
                color: white;
                background: transparent;
                border: none;
                font-family: 'Inter';
                font-size: 13px;
            }
        """)
        self.input_field.returnPressed.connect(self.handle_action)
        self.input_field.textChanged.connect(self.update_button_icon)
        self.layout.addWidget(self.input_field)

        # Action Button (Mic / Send)
        self.send_btn = QPushButton(".ılı.")
        self.send_btn.setFixedSize(36, 36)
        self.send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.send_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 255, 255, 100);
                color: black;
                border-radius: 18px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: rgba(0, 255, 255, 255);
            }
        """)
        self.send_btn.clicked.connect(self.handle_action)
        self.layout.addWidget(self.send_btn)

    def update_button_icon(self, text):
        if text.strip():
            self.send_btn.setText("➔")
            self.send_btn.setStyleSheet(self.send_btn.styleSheet().replace("100", "200"))
        else:
            self.send_btn.setText(".ılı." if not self.is_recording else "●")
            self.send_btn.setStyleSheet(self.send_btn.styleSheet().replace("200", "100"))

    def handle_action(self):
        text = self.input_field.text().strip()
        if text:
            # Send Message Mode
            self.message_sent.emit(text)
            self.input_field.clear()
            self.is_recording = False
        else:
            # Voice Memo Toggle Mode
            self.is_recording = not self.is_recording
            self.status = "LISTENING" if self.is_recording else "STATIC"
            self.update_button_icon("")

    def update_pulse(self):
        import math
        import time
        self.pulse_val = (math.sin(time.time() * 3) + 1) / 2
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 1. Pill Background
        painter.setBrush(QColor(40, 40, 40, 230))
        painter.setPen(QPen(QColor(255, 255, 255, 40), 1))
        rect = QRect(0, 0, self.width(), self.height())
        painter.drawRoundedRect(rect.adjusted(1,1,-1,-1), 30, 30)
        
        # 2. AI Health Heartbeat Logic
        # Colors: Healthy=Cyan, Listening=Yellow, Thinking=Orange, Speaking=Purple, Error=Red
        color = QColor(0, 255, 255) # Cyan (Default Healthy)
        pulse_speed = 3 # Default calm pulse
        
        if self.status == "LISTENING":
            color = QColor(255, 255, 0) # Yellow
            pulse_speed = 4 # Slightly faster
        elif self.status == "THINKING":
            color = QColor(255, 165, 0)
            pulse_speed = 8 # Fast
        elif self.status == "SPEAKING":
            color = QColor(138, 43, 226)
            pulse_speed = 5 # Medium
        elif self.status == "ERROR":
            color = QColor(255, 50, 50)
            pulse_speed = 12 # Rapid alert
            
        # Update pulse calculation with variable speed
        import math, time
        val = (math.sin(time.time() * pulse_speed) + 1) / 2
        
        pulse_alpha = int(100 + 155 * val)
        radius = 7 + 3 * val
        
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), pulse_alpha))
        painter.setPen(Qt.PenStyle.NoPen)
        
        center_x = 15 + 12
        center_y = self.height() // 2
        painter.drawEllipse(QPoint(int(center_x), int(center_y)), int(radius), int(radius))

class ChatAssistantWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        # Main Layout (Vertical)
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # 1. Chat Container (Glassy Panel)
        self.chat_container = QFrame()
        self.chat_container.setStyleSheet("""
            QFrame {
                background-color: rgba(25, 25, 25, 230);
                border-top-left-radius: 24px;
                border-top-right-radius: 24px;
                border: 1px solid rgba(255, 255, 255, 40);
            }
        """)
        self.chat_layout = QVBoxLayout(self.chat_container)
        
        # Scroll Area for Messages
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("background: transparent; border: none;")
        self.scroll.verticalScrollBar().setStyleSheet("QScrollBar { width: 0px; }") 
        
        self.scroll_content = QWidget()
        self.scroll_content.setStyleSheet("background: transparent;")
        self.messages_layout = QVBoxLayout(self.scroll_content)
        self.messages_layout.addStretch()
        
        self.scroll.setWidget(self.scroll_content)
        self.chat_layout.addWidget(self.scroll)
        self.main_layout.addWidget(self.chat_container)

        # 2. Input Container (Bottom Pill)
        self.input_container = QFrame()
        self.input_container.setFixedHeight(100)
        self.input_container.setStyleSheet("""
            QFrame {
                background-color: rgba(25, 25, 25, 240);
                border-bottom-left-radius: 24px;
                border-bottom-right-radius: 24px;
                border-top: 1px solid rgba(255, 255, 255, 20);
                border-left: 1px solid rgba(255, 255, 255, 40);
                border-right: 1px solid rgba(255, 255, 255, 40);
                border-bottom: 1px solid rgba(255, 255, 255, 40);
            }
        """)
        input_layout = QVBoxLayout(self.input_container)
        input_layout.setContentsMargins(20, 0, 20, 20)
        
        self.pill = VoicePill()
        self.pill.message_sent.connect(self.handle_new_message)
        input_layout.addWidget(self.pill, alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.main_layout.addWidget(self.input_container)

        # Demo Messages
        self.add_message("Surgeon: Show last chest X-ray.", True)
        self.add_message("AI Assistant: Loading X-ray for Patient ID 4402. Displaying on main monitor.", False)

    def add_message(self, text, is_user=True):
        bubble = ChatBubble(text, is_user)
        self.messages_layout.insertWidget(self.messages_layout.count() - 1, bubble)
        # Safety check: ensure widget hasn't been deleted
        QTimer.singleShot(100, lambda: self.safe_scroll())

    def safe_scroll(self):
        try:
            if hasattr(self, 'scroll') and self.scroll and not self.scroll.isHidden():
                self.scroll.verticalScrollBar().setValue(
                    self.scroll.verticalScrollBar().maximum()
                )
        except RuntimeError:
            pass # Object was deleted

    def handle_new_message(self, text):
        self.add_message(f"Surgeon: {text}", True)
        QTimer.singleShot(1000, lambda: self.add_message("AI Assistant: Processing your request...", False))

    def set_progress(self, val):
        self.pill.progress = val
        self.pill.update()

class ZeroTouchAudioAssist(QMainWindow):
    exit_requested = pyqtSignal()
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Window Properties
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | 
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Dimensions
        self.win_w, self.win_h = 400, 600
        self.resize(self.win_w, self.win_h)
        
        # Position at bottom-right (Margins: 10, 10)
        screen = QApplication.primaryScreen().availableGeometry()
        margin_x, margin_y = 10, 10
        self.move(screen.width() - self.win_w - margin_x, screen.height() - self.win_h - margin_y)
        
        # Add the Modular Chat Widget
        self.chat_widget = ChatAssistantWidget()
        self.setCentralWidget(self.chat_widget)

        # Exit Button (Top Right)
        self.exit_btn = QPushButton("✕", self)
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
        
        self.show()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    from PyQt6.QtGui import QFont
    font = QFont("Inter")
    if not font.exactMatch(): font = QFont("Segoe UI", 10)
    app.setFont(font)
    window = ZeroTouchAudioAssist()
    sys.exit(app.exec())
