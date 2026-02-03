import numpy as np
import soundfile as sf
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QPainter, QColor, QBrush, QPen

class WaveformWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.audio_path = None
        self.peaks = None
        self.setMinimumHeight(60)
        self.setFixedHeight(100)
        
        # Style
        self.bar_color = QColor(0, 120, 215)  # Accent color
        self.bg_color = Qt.transparent
        self.bar_width = 3
        self.bar_gap = 1

    def load_file(self, file_path):
        """Load audio file and calculate waveform peaks."""
        self.audio_path = file_path
        self.peaks = None
        
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
            # We want roughly self.width() / (bar_width + gap) bars
            # But width is dynamic. Let's compute a fixed number of bars for now
            # or compute peaks based on a fixed resolution (e.g. 200 bars)
            num_bars = 200
            chunk_size = len(data) // num_bars
            if chunk_size < 1:
                chunk_size = 1
                
            # Calculate RMS or max for each chunk
            # Reshape into chunks (truncate leftover)
            num_chunks = len(data) // chunk_size
            data = data[:num_chunks * chunk_size]
            chunks = data.reshape(num_chunks, chunk_size)
            
            # Use max amplitude per chunk
            self.peaks = np.max(np.abs(chunks), axis=1)
            
            self.update() # Trigger repaint
            
        except Exception as e:
            print(f"Error loading waveform: {e}")
            self.peaks = None
            self.update()

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
            
        # Calculate bar width dynamically to fit width
        # total_width = count * (bar_w + gap)
        # We can stretch the bars to fit
        step = w / count
        bar_w = max(1, step - 1)
        
        painter.setBrush(QBrush(self.bar_color))
        painter.setPen(Qt.NoPen)
        
        # Center line
        mid_y = h / 2
        
        for i, peak in enumerate(self.peaks):
            x = i * step
            # Height based on peak (0.0 to 1.0)
            # Scale so max peak fills height
            bar_h = peak * h
            
            # Draw centered vertically
            y = mid_y - (bar_h / 2)
            
            rect = QRectF(x, y, bar_w, bar_h)
            painter.drawRoundedRect(rect, 1, 1)

