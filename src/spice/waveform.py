import numpy as np
import soundfile as sf
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QRectF, pyqtSignal
from PyQt5.QtGui import QPainter, QColor, QBrush, QPen

class WaveformWidget(QWidget):
    seek_requested = pyqtSignal(float)  # Signal emitted with percentage (0.0 to 1.0)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.audio_path = None
        self.peaks = None
        self.play_progress = 0.0  # 0.0 to 1.0
        self.setMinimumHeight(60)
        self.setFixedHeight(100)
        self.setCursor(Qt.PointingHandCursor)
        
        # Style
        self.bar_color = QColor(0, 120, 215)  # Accent color
        self.played_color = QColor(255, 255, 255, 150) # Highlight for played part
        self.bg_color = Qt.transparent
        self.playhead_color = QColor(255, 255, 255)
        self.bar_width = 3
        self.bar_gap = 1

    def load_file(self, file_path):
        """Load audio file and calculate waveform peaks."""
        self.audio_path = file_path
        self.peaks = None
        self.play_progress = 0.0
        
        try:
            # Read audio
            data, samplerate = sf.read(file_path)
            
            # Convert to mono if stereo
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
                
            # Normalize
            max_val = np.max(np.abs(data))
            if max_val > 0:
                data = data / max_val
                
            # Downsample for visualization
            num_bars = 200
            chunk_size = len(data) // num_bars
            if chunk_size < 1:
                chunk_size = 1
                
            # Calculate max for each chunk
            num_chunks = len(data) // chunk_size
            data = data[:num_chunks * chunk_size]
            chunks = data.reshape(num_chunks, chunk_size)
            
            self.peaks = np.max(np.abs(chunks), axis=1)
            
            self.update() # Trigger repaint
            
        except Exception as e:
            print(f"Error loading waveform: {e}")
            self.peaks = None
            self.update()

    def set_progress(self, progress):
        """Update the playhead position (0.0 to 1.0)."""
        self.play_progress = max(0.0, min(1.0, progress))
        self.update()

    def mousePressEvent(self, event):
        if self.peaks is not None:
            progress = event.x() / self.width()
            self.seek_requested.emit(progress)
            self.set_progress(progress)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw background (transparent)
        painter.fillRect(self.rect(), self.bg_color)
        
        if self.peaks is None:
            painter.setPen(Qt.gray)
            painter.drawText(self.rect(), Qt.AlignCenter, "No Audio Selected")
            return

        # Draw bars
        w = self.width()
        h = self.height()
        
        count = len(self.peaks)
        if count == 0:
            return
            
        step = w / count
        bar_w = max(1, step - 1)
        
        # Center line
        mid_y = h / 2
        
        playhead_x = self.play_progress * w

        for i, peak in enumerate(self.peaks):
            x = i * step
            bar_h = peak * h
            y = mid_y - (bar_h / 2)
            
            rect = QRectF(x, y, bar_w, bar_h)
            
            # Color bars based on whether they've been played
            if x < playhead_x:
                painter.setBrush(QBrush(self.bar_color))
            else:
                painter.setBrush(QBrush(self.bar_color.lighter(150)))
                painter.setOpacity(0.5)
            
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(rect, 1, 1)
            painter.setOpacity(1.0)

        # Draw playhead
        painter.setPen(QPen(self.playhead_color, 2))
        painter.drawLine(int(playhead_x), 0, int(playhead_x), h)

