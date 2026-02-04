from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout
from qfluentwidgets import PrimaryPushButton, FluentIcon as FIF, SubtitleLabel, BodyLabel

class OnboardingView(QWidget):
    """
    The 'Zero State' view shown when the library is empty.
    """
    add_folder_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(10)
        
        # Big Icon / Illustration Placeholder
        # Using a large emoji for now, or a FluentIcon if preferred
        self.icon_label = QLabel("📂", self)
        self.icon_label.setStyleSheet("font-size: 80px; margin-bottom: 10px;")
        self.icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.icon_label)
        
        # Headline
        self.title = SubtitleLabel("Your library is empty", self)
        self.title.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title)
        
        # Subtext
        self.subtext = BodyLabel("Drop your first sample pack to start searching by vibe, BPM, and Key.", self)
        self.subtext.setAlignment(Qt.AlignCenter)
        self.subtext.setStyleSheet("color: gray;")
        layout.addWidget(self.subtext)
        
        layout.addSpacing(20)
        
        # Action Button
        self.add_btn = PrimaryPushButton(FIF.ADD, "Add Sample Folder", self)
        self.add_btn.setFixedWidth(200)
        self.add_btn.clicked.connect(self.add_folder_requested.emit)
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch(1)
        btn_layout.addWidget(self.add_btn)
        btn_layout.addStretch(1)
        layout.addLayout(btn_layout)
        
        # Extra Tip
        self.tip_label = BodyLabel("Tip: You can drag and drop folders directly here (soon!)", self)
        self.tip_label.setAlignment(Qt.AlignCenter)
        self.tip_label.setStyleSheet("font-size: 12px; color: #888; margin-top: 30px;")
        # layout.addWidget(self.tip_label)
