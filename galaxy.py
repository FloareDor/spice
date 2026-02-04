"""
LocalVibe Galaxy Map - UMAP-based sample visualization
Plots all samples in 2D space based on their semantic embeddings.
"""
import json
from pathlib import Path
from typing import Optional, Callable

import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPointF, QRectF
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QFont, QWheelEvent, QMouseEvent

import database as db


class UMAPThread(QThread):
    """Background thread for computing UMAP projection."""
    finished = pyqtSignal(np.ndarray, list)  # coords, metadata list
    progress = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, db_path: str):
        super().__init__()
        self.db_path = db_path

    def run(self):
        try:
            self.progress.emit("Loading embeddings from database...")

            # Get all samples from database
            lance_db = db.get_db(self.db_path)
            table = db.create_samples_table(lance_db)

            # Fetch all rows
            rows = table.search().limit(100000).to_list()

            if len(rows) < 5:
                self.error.emit("Need at least 5 samples for visualization")
                return

            self.progress.emit(f"Processing {len(rows)} samples...")

            # Extract embeddings and metadata
            embeddings = []
            metadata = []
            for row in rows:
                embeddings.append(np.array(row["vector"], dtype=np.float32))
                meta = json.loads(row["metadata"])
                metadata.append(meta)

            embeddings = np.array(embeddings)

            self.progress.emit("Computing UMAP projection (this may take a moment)...")

            # Import UMAP here to avoid slow import at startup
            import umap

            # Configure UMAP for good visualization
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(15, len(embeddings) - 1),
                min_dist=0.1,
                metric='cosine',
                random_state=42,
                low_memory=True,
            )

            coords = reducer.fit_transform(embeddings)

            # Normalize to 0-1 range
            coords_min = coords.min(axis=0)
            coords_max = coords.max(axis=0)
            coords_range = coords_max - coords_min
            coords_range[coords_range == 0] = 1  # Avoid division by zero
            coords = (coords - coords_min) / coords_range

            self.progress.emit("Done!")
            self.finished.emit(coords, metadata)

        except Exception as e:
            self.error.emit(f"UMAP error: {str(e)}")


class GalaxyWidget(QWidget):
    """
    Interactive 2D scatter plot of all samples based on UMAP projection.
    Supports zooming, panning, and clicking to select samples.
    """
    sample_selected = pyqtSignal(dict)  # Emits metadata of selected sample
    sample_hovered = pyqtSignal(dict)   # Emits metadata of hovered sample

    # Color palette for different sample types (based on tags)
    COLORS = {
        "Kick": QColor(220, 60, 60),      # Red
        "Snare": QColor(60, 140, 220),    # Blue
        "Hi-Hat": QColor(220, 180, 60),   # Yellow
        "Clap": QColor(180, 60, 220),     # Purple
        "Bass": QColor(60, 180, 100),     # Green
        "Synth": QColor(100, 200, 220),   # Cyan
        "Pad": QColor(160, 120, 200),     # Lavender
        "Vocal": QColor(240, 140, 100),   # Orange
        "FX": QColor(140, 140, 140),      # Gray
        "default": QColor(100, 120, 180), # Default blue-gray
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 400)
        self.setMouseTracking(True)

        # Data
        self.coords: Optional[np.ndarray] = None
        self.metadata: list = []
        self.highlighted_indices: set = set()  # Indices of highlighted (search result) samples

        # View state
        self.zoom = 1.0
        self.pan_offset = QPointF(0, 0)
        self.dragging = False
        self.last_mouse_pos = QPointF()

        # Interaction state
        self.hovered_index: Optional[int] = None
        self.selected_index: Optional[int] = None

        # Loading state
        self.loading = False
        self.status_message = "Click 'Load Galaxy' to visualize your samples"

        # Style
        self.dot_radius = 4
        self.highlight_radius = 8
        self.setStyleSheet("background-color: #1a1a2e;")

    def load_data(self, db_path: str = "./localvibe.lance"):
        """Start loading and computing UMAP projection."""
        self.loading = True
        self.status_message = "Loading..."
        self.update()

        self.umap_thread = UMAPThread(db_path)
        self.umap_thread.finished.connect(self.on_umap_finished)
        self.umap_thread.progress.connect(self.on_progress)
        self.umap_thread.error.connect(self.on_error)
        self.umap_thread.start()

    def on_umap_finished(self, coords: np.ndarray, metadata: list):
        """Handle completed UMAP computation."""
        self.coords = coords
        self.metadata = metadata
        self.loading = False
        self.status_message = f"{len(metadata)} samples loaded"

        # Reset view
        self.zoom = 1.0
        self.pan_offset = QPointF(0, 0)

        self.update()

    def on_progress(self, message: str):
        """Handle progress updates."""
        self.status_message = message
        self.update()

    def on_error(self, message: str):
        """Handle errors."""
        self.loading = False
        self.status_message = message
        self.update()

    def highlight_samples(self, file_paths: set):
        """Highlight specific samples (e.g., search results)."""
        self.highlighted_indices.clear()
        if self.metadata:
            for i, meta in enumerate(self.metadata):
                if meta.get("path") in file_paths:
                    self.highlighted_indices.add(i)
        self.update()

    def clear_highlights(self):
        """Clear all highlights."""
        self.highlighted_indices.clear()
        self.update()

    def get_color_for_sample(self, meta: dict) -> QColor:
        """Determine color based on sample tags."""
        tags = meta.get("tags", [])
        for tag in tags:
            for key, color in self.COLORS.items():
                if key.lower() in tag.lower():
                    return color
        return self.COLORS["default"]

    def world_to_screen(self, x: float, y: float) -> QPointF:
        """Convert world coordinates (0-1) to screen coordinates."""
        margin = 50
        w = self.width() - 2 * margin
        h = self.height() - 2 * margin

        # Apply zoom and pan
        screen_x = margin + (x * w * self.zoom) + self.pan_offset.x()
        screen_y = margin + ((1 - y) * h * self.zoom) + self.pan_offset.y()  # Flip Y

        return QPointF(screen_x, screen_y)

    def screen_to_world(self, screen_x: float, screen_y: float) -> tuple:
        """Convert screen coordinates to world coordinates (0-1)."""
        margin = 50
        w = self.width() - 2 * margin
        h = self.height() - 2 * margin

        x = (screen_x - margin - self.pan_offset.x()) / (w * self.zoom)
        y = 1 - (screen_y - margin - self.pan_offset.y()) / (h * self.zoom)

        return (x, y)

    def find_sample_at(self, pos: QPointF) -> Optional[int]:
        """Find the sample index at the given screen position."""
        if self.coords is None:
            return None

        click_radius = 10  # Pixels

        for i, (x, y) in enumerate(self.coords):
            screen_pos = self.world_to_screen(x, y)
            dx = pos.x() - screen_pos.x()
            dy = pos.y() - screen_pos.y()
            if dx * dx + dy * dy < click_radius * click_radius:
                return i

        return None

    def paintEvent(self, event):
        """Draw the galaxy map."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Background
        painter.fillRect(self.rect(), QColor(26, 26, 46))

        # Status message
        if self.loading or self.coords is None:
            painter.setPen(QColor(150, 150, 180))
            painter.setFont(QFont("Segoe UI", 12))
            painter.drawText(self.rect(), Qt.AlignCenter, self.status_message)
            return

        # Draw all dots
        for i, (x, y) in enumerate(self.coords):
            screen_pos = self.world_to_screen(x, y)

            # Skip if outside viewport
            if not self.rect().contains(screen_pos.toPoint()):
                continue

            meta = self.metadata[i]
            color = self.get_color_for_sample(meta)

            # Determine radius and style
            if i == self.selected_index:
                radius = self.highlight_radius
                painter.setPen(QPen(QColor(255, 255, 255), 2))
                painter.setBrush(QBrush(color))
            elif i in self.highlighted_indices:
                radius = self.highlight_radius - 2
                painter.setPen(QPen(color.lighter(150), 2))
                painter.setBrush(QBrush(color))
            elif i == self.hovered_index:
                radius = self.dot_radius + 2
                painter.setPen(QPen(color.lighter(150), 1))
                painter.setBrush(QBrush(color.lighter(130)))
            else:
                radius = self.dot_radius
                painter.setPen(Qt.NoPen)
                darker = QColor(color)
                darker.setAlpha(180)
                painter.setBrush(QBrush(darker))

            painter.drawEllipse(screen_pos, radius, radius)

        # Draw tooltip for hovered sample
        if self.hovered_index is not None:
            meta = self.metadata[self.hovered_index]
            x, y = self.coords[self.hovered_index]
            screen_pos = self.world_to_screen(x, y)

            filename = meta.get("filename", "Unknown")
            bpm = meta.get("bpm", 0)
            key = meta.get("key", "")
            tags = meta.get("tags", [])[:3]

            tooltip = f"{filename}\n{bpm:.0f} BPM  {key}"
            if tags:
                tooltip += f"\n{', '.join(tags)}"

            # Draw tooltip background
            font = QFont("Segoe UI", 9)
            painter.setFont(font)
            metrics = painter.fontMetrics()
            lines = tooltip.split('\n')
            max_width = max(metrics.horizontalAdvance(line) for line in lines)
            height = len(lines) * metrics.height()

            tooltip_rect = QRectF(
                screen_pos.x() + 15,
                screen_pos.y() - height / 2,
                max_width + 16,
                height + 8
            )

            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(40, 40, 60, 230)))
            painter.drawRoundedRect(tooltip_rect, 4, 4)

            painter.setPen(QColor(220, 220, 240))
            y_offset = tooltip_rect.top() + metrics.ascent() + 4
            for line in lines:
                painter.drawText(QPointF(tooltip_rect.left() + 8, y_offset), line)
                y_offset += metrics.height()

        # Draw sample count
        painter.setPen(QColor(100, 100, 130))
        painter.setFont(QFont("Segoe UI", 9))
        painter.drawText(10, self.height() - 10, self.status_message)

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press for selection and panning."""
        if event.button() == Qt.LeftButton:
            # Check if clicking on a sample
            idx = self.find_sample_at(event.pos())
            if idx is not None:
                self.selected_index = idx
                self.sample_selected.emit(self.metadata[idx])
                self.update()
            else:
                # Start panning
                self.dragging = True
                self.last_mouse_pos = event.pos()

        elif event.button() == Qt.RightButton:
            # Reset view
            self.zoom = 1.0
            self.pan_offset = QPointF(0, 0)
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move for hovering and panning."""
        if self.dragging:
            delta = event.pos() - self.last_mouse_pos
            self.pan_offset += QPointF(delta.x(), delta.y())
            self.last_mouse_pos = event.pos()
            self.update()
        else:
            # Update hover state
            idx = self.find_sample_at(event.pos())
            if idx != self.hovered_index:
                self.hovered_index = idx
                if idx is not None:
                    self.sample_hovered.emit(self.metadata[idx])
                self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release."""
        if event.button() == Qt.LeftButton:
            self.dragging = False

    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zooming."""
        # Zoom centered on mouse position
        mouse_pos = event.pos()
        old_world = self.screen_to_world(mouse_pos.x(), mouse_pos.y())

        # Adjust zoom
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom *= 1.15
        else:
            self.zoom /= 1.15

        # Clamp zoom
        self.zoom = max(0.5, min(10.0, self.zoom))

        # Adjust pan to keep mouse position fixed
        new_screen = self.world_to_screen(old_world[0], old_world[1])
        self.pan_offset += QPointF(mouse_pos.x() - new_screen.x(),
                                   mouse_pos.y() - new_screen.y())

        self.update()
