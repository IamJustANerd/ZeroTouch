import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                             QWidget, QCheckBox, QLabel, QHBoxLayout, QFrame, QComboBox,
                             QStackedWidget, QLineEdit, QFormLayout)
from PyQt6.QtCore import Qt, QPoint, QRect
from PyQt6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPen
from overlay_pyqt import ZeroTouchOverlay
from audio_assist_ui import ZeroTouchAudioAssist

class ZeroTouchLauncher(QMainWindow):
    def __init__(self):
        super().__init__()
        self._drag_pos = QPoint()
        self.initUI()

    def initUI(self):
        # Window Properties
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(400, 350)
        
        # Main Stacked Widget
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        
        # --- PAGE 1: MAIN MENU ---
        self.main_page = QFrame()
        self.stack.addWidget(self.main_page)
        
        main_layout = QVBoxLayout(self.main_page)
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(20)

        # Settings Wheel (Top Left)
        settings_btn = QPushButton("⚙", self.main_page)
        settings_btn.setGeometry(15, 10, 40, 40)
        settings_btn.setStyleSheet("""
            QPushButton { 
                color: rgba(255, 255, 255, 150); 
                border: none; 
                font-size: 24px; 
                background: transparent;
            }
            QPushButton:hover { color: #00FFFF; }
        """)
        settings_btn.clicked.connect(lambda: self.stack.setCurrentIndex(1))

        # Close Button (Top Right)
        close_btn = QPushButton("✕", self.main_page)
        close_btn.setGeometry(360, 10, 30, 30)
        close_btn.setStyleSheet("""
            QPushButton { 
                color: rgba(255, 255, 255, 150); 
                border: none; 
                font-size: 18px; 
                background: transparent;
            }
            QPushButton:hover { color: #FF4B4B; }
        """)
        close_btn.clicked.connect(self.close)

        # Title Section
        self.title_label = QLabel("ZeroTouch")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("font-size: 38px; font-weight: bold; color: #00FFFF; background: transparent;")
        main_layout.addWidget(self.title_label)

        self.sub_label = QLabel("Operating Room Interface")
        self.sub_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sub_label.setStyleSheet("font-size: 14px; color: rgba(255, 255, 255, 180); letter-spacing: 2px; background: transparent;")
        main_layout.addWidget(self.sub_label)

        main_layout.addStretch()

        # Mode Selection Dropdown
        mode_label = QLabel("OPERATIONAL MODE")
        mode_label.setStyleSheet("font-size: 10px; color: rgba(255, 255, 255, 120); letter-spacing: 1px;")
        main_layout.addWidget(mode_label)

        self.mode_box = QComboBox()
        self.mode_box.addItems(["Camera & Audio Assistance", "Camera Assist. Only", "Audio Assist. Only"])
        self.mode_box.setMinimumHeight(40)
        self.mode_box.setStyleSheet("""
            QComboBox { background: rgba(255, 255, 255, 15); border: 1px solid rgba(255, 255, 255, 40); border-radius: 8px; padding-left: 15px; color: white; font-size: 13px; }
            QComboBox::drop-down { border: none; width: 30px; }
            QComboBox::down-arrow { image: none; border-left: 5px solid transparent; border-right: 5px solid transparent; border-top: 5px solid #00FFFF; margin-right: 15px; }
            QComboBox QAbstractItemView { background-color: #1A1A1A; color: white; selection-background-color: #00FFFF; selection-color: black; outline: none; border: 1px solid rgba(255, 255, 255, 40); }
        """)
        main_layout.addWidget(self.mode_box)

        # Debug Toggle
        self.debug_check = QCheckBox("Enable Debug Mode (Show Camera)")
        self.debug_check.setChecked(True)
        self.debug_check.setStyleSheet("""
            QCheckBox { color: rgba(255, 255, 255, 200); font-size: 13px; background: transparent; }
            QCheckBox::indicator { width: 18px; height: 18px; border: 2px solid rgba(255, 255, 255, 100); border-radius: 4px; }
            QCheckBox::indicator:checked { background-color: #00FFFF; border-color: #00FFFF; }
        """)
        main_layout.addWidget(self.debug_check)

        # Start Button
        self.start_btn = QPushButton("START ENGINE")
        self.start_btn.setMinimumHeight(60)
        self.start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_btn.setStyleSheet("""
            QPushButton { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00FFFF, stop:1 #008080); color: black; border-radius: 12px; font-size: 16px; font-weight: bold; letter-spacing: 1px; }
            QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00FFFF, stop:1 #00AAAA); }
        """)
        self.start_btn.clicked.connect(self.launch_app)
        main_layout.addWidget(self.start_btn)

        # --- PAGE 2: SETTINGS ---
        self.settings_page = QFrame()
        self.stack.addWidget(self.settings_page)
        
        settings_layout = QVBoxLayout(self.settings_page)
        settings_layout.setContentsMargins(40, 40, 40, 40)
        
        # Back Button
        back_btn = QPushButton("← BACK", self.settings_page)
        back_btn.setStyleSheet("color: #00FFFF; border: none; font-weight: bold; background: transparent;")
        back_btn.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        settings_layout.addWidget(back_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        
        settings_title = QLabel("SETTINGS")
        settings_title.setStyleSheet("font-size: 24px; font-weight: bold; color: white; background: transparent;")
        settings_layout.addWidget(settings_title, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Placeholder Form
        form_widget = QWidget()
        form_layout = QFormLayout(form_widget)
        form_layout.setSpacing(15)
        
        field_style = "background: rgba(255,255,255,20); border: 1px solid rgba(255,255,255,40); color: white; padding: 5px; border-radius: 4px;"
        
        self.cam_id = QLineEdit("0")
        self.cam_id.setStyleSheet(field_style)
        form_layout.addRow(QLabel("Camera ID:"), self.cam_id)
        
        self.threshold = QLineEdit("0.7")
        self.threshold.setStyleSheet(field_style)
        form_layout.addRow(QLabel("Confidence:"), self.threshold)
        
        for label in form_widget.findChildren(QLabel):
            label.setStyleSheet("color: rgba(255,255,255,180); font-size: 12px;")
            
        settings_layout.addWidget(form_widget)
        settings_layout.addStretch()
        
        save_btn = QPushButton("SAVE SETTINGS")
        save_btn.setMinimumHeight(40)
        save_btn.setStyleSheet("background: #00FFFF; color: black; font-weight: bold; border-radius: 8px;")
        save_btn.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        settings_layout.addWidget(save_btn)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Glass BG
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor(30, 30, 30, 240))
        gradient.setColorAt(1, QColor(10, 10, 10, 240))
        
        painter.setBrush(gradient)
        painter.setPen(QPen(QColor(255, 255, 255, 40), 2))
        
        rect = QRect(0, 0, self.width(), self.height())
        painter.drawRoundedRect(rect.adjusted(1,1,-1,-1), 24, 24)

    # Window Drag Logic
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def launch_app(self):
        mode = self.mode_box.currentText()
        debug_mode = self.debug_check.isChecked()
        print(f"--- Launching Engine: {mode} (Debug: {debug_mode}) ---")
        self.hide()
        
        try:
            if mode == "Audio Assist. Only":
                print("Initializing Audio Assist UI...")
                self.overlay = ZeroTouchAudioAssist()
            elif mode == "Camera & Audio Assistance":
                print("Initializing Combined Camera + Audio UI...")
                self.overlay = ZeroTouchOverlay(debug=debug_mode, show_chat=True)
            else:
                print("Initializing Camera Overlay UI...")
                self.overlay = ZeroTouchOverlay(debug=debug_mode, show_chat=False)
            
            # Connect Exit Signal
            self.overlay.exit_requested.connect(self.return_to_menu)
            print("Engine running. Check bottom-right of screen.")
        except Exception as e:
            print(f"CRITICAL ERROR during launch: {e}")
            self.show()

    def return_to_menu(self):
        print("Returning to menu...")
        if hasattr(self, 'overlay') and self.overlay:
            self.overlay.close()
            self.overlay = None
        self.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    launcher = ZeroTouchLauncher()
    launcher.show()
    sys.exit(app.exec())
