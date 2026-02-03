import sys
import os
from pathlib import Path

from PyQt5.QtCore import Qt, QUrl, QMimeData, QSize, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QListWidgetItem, QListWidget, QWidget, QVBoxLayout, 
    QHBoxLayout, QLabel, QFileDialog, QProgressBar, QTextEdit
)
from PyQt5.QtGui import QDrag, QIcon
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

from qfluentwidgets import (
    FluentWindow, ListWidget, FluentIcon as FIF, SearchLineEdit, 
    PushButton, PrimaryPushButton, SubtitleLabel
)

import search
import indexer
from waveform import WaveformWidget

# --- Threads ---

class SearchThread(QThread):
    results_ready = pyqtSignal(list)

    def __init__(self, query_text):
        super().__init__()
        self.query_text = query_text

    def run(self):
        try:
            results = search.search_by_text(self.query_text, "./localvibe.lance", limit=50)
            self.results_ready.emit(results)
        except Exception as e:
            print(f"Search thread error: {e}")
            self.results_ready.emit([])

class IndexingThread(QThread):
    progress_signal = pyqtSignal(int, int, str) # processed, total, message
    finished_signal = pyqtSignal(dict)
    
    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path
        
    def run(self):
        try:
            stats = indexer.index_folder(
                Path(self.folder_path),
                db_path="./localvibe.lance",
                recursive=True,
                progress_callback=self.emit_progress
            )
            self.finished_signal.emit(stats)
        except Exception as e:
            self.progress_signal.emit(0, 0, f"Error: {e}")
            self.finished_signal.emit({})

    def emit_progress(self, processed, total, msg):
        self.progress_signal.emit(processed, total, msg)

# --- Widgets ---

class SampleListWidget(ListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setSelectionMode(QListWidget.SingleSelection)
        self.setObjectName("sampleList")

    def startDrag(self, supportedActions):
        item = self.currentItem()
        if not item:
            return
        
        file_path = item.data(Qt.UserRole)
        if not file_path:
            return

        mime = QMimeData()
        url = QUrl.fromLocalFile(file_path)
        mime.setUrls([url])
        
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec_(Qt.CopyAction)

class LibraryInterface(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("libraryInterface")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)
        
        # Header
        title = SubtitleLabel("Library Management", self)
        layout.addWidget(title)
        
        # Controls
        btn_layout = QHBoxLayout()
        self.add_btn = PrimaryPushButton(FIF.ADD, "Add Folder to Index", self)
        self.add_btn.clicked.connect(self.add_folder)
        btn_layout.addWidget(self.add_btn)
        btn_layout.addStretch(1)
        layout.addLayout(btn_layout)
        
        # Progress
        self.status_label = QLabel("Ready", self)
        layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        
        # Log
        self.log_area = QTextEdit(self)
        self.log_area.setReadOnly(True)
        self.log_area.setPlaceholderText("Indexing logs will appear here...")
        layout.addWidget(self.log_area)
        
    def add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Sample Folder")
        if not folder:
            return
            
        self.log_area.append(f"Selected: {folder}")
        self.start_indexing(folder)
        
    def start_indexing(self, folder):
        self.add_btn.setEnabled(False)
        self.progress_bar.show()
        self.progress_bar.setValue(0)
        
        self.thread = IndexingThread(folder)
        self.thread.progress_signal.connect(self.update_progress)
        self.thread.finished_signal.connect(self.indexing_finished)
        self.thread.start()
        
    def update_progress(self, processed, total, msg):
        self.status_label.setText(msg)
        self.log_area.append(msg)
        if total > 0:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(processed)
            
    def indexing_finished(self, stats):
        self.add_btn.setEnabled(True)
        self.progress_bar.hide()
        
        if stats:
            summary = (
                f"\nIndexing Complete!\n"
                f"Total: {stats.get('total', 0)}\n"
                f"New: {stats.get('new', 0)}\n"
                f"Skipped: {stats.get('skipped', 0)}\n"
                f"Errors: {stats.get('errors', 0)}\n"
            )
            self.log_area.append(summary)
            self.status_label.setText("Done.")
        else:
            self.status_label.setText("Indexing failed.")


class MainWindow(FluentWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LocalVibe")
        self.resize(1000, 750)
        
        # Center window
        desktop = QApplication.primaryScreen().availableGeometry()
        w, h = desktop.width(), desktop.height()
        self.move(w//2 - self.width()//2, h//2 - self.height()//2)
        
        # --- Samples Page ---
        self.home_widget = QWidget()
        self.home_widget.setObjectName("homeWidget")
        self.home_layout = QVBoxLayout(self.home_widget)
        
        # Search
        self.search_bar = SearchLineEdit(self)
        self.search_bar.setPlaceholderText("Search sounds (e.g. 'dark bass', '140 bpm')")
        self.search_bar.returnPressed.connect(self.perform_search)
        self.home_layout.addWidget(self.search_bar)
        
        # List
        self.sample_list = SampleListWidget(self)
        self.sample_list.itemClicked.connect(self.on_sample_clicked)
        self.home_layout.addWidget(self.sample_list)
        
        # Waveform & Info
        self.info_layout = QVBoxLayout()
        self.info_label = QLabel("Select a sample to preview")
        self.info_label.setObjectName("infoLabel")
        self.info_layout.addWidget(self.info_label)
        
        self.waveform = WaveformWidget(self)
        self.info_layout.addWidget(self.waveform)
        
        self.home_layout.addLayout(self.info_layout)
        
        # --- Library Page ---
        self.library_interface = LibraryInterface(self)
        
        # --- Navigation ---
        self.addSubInterface(self.home_widget, icon=FIF.MUSIC, text='Samples')
        self.addSubInterface(self.library_interface, icon=FIF.FOLDER, text='Library')
        
        # Media Player
        self.player = QMediaPlayer()

    def perform_search(self):
        query = self.search_bar.text().strip()
        if not query:
            return
            
        self.search_bar.setEnabled(False) 
        
        self.search_thread = SearchThread(query)
        self.search_thread.results_ready.connect(self.handle_results)
        self.search_thread.finished.connect(lambda: self.search_bar.setEnabled(True))
        self.search_thread.start()

    def handle_results(self, results):
        self.sample_list.clear()
        
        if not results:
            item = QListWidgetItem("No results found.")
            item.setFlags(Qt.NoItemFlags)
            self.sample_list.addItem(item)
            return

        for r in results:
            meta = r.get('metadata', {})
            filename = meta.get('filename', 'Unknown')
            file_path = meta.get('path', '')
            bpm = meta.get('bpm', 0)
            key = meta.get('key', '')
            tags = meta.get('tags', [])
            
            # Format tags
            tag_str = ""
            if tags:
                # Top 3 tags
                tag_str = " | ".join(tags[:3])
                tag_str = f"   ({tag_str})"
            
            display_text = f"{filename}   [{bpm:.1f} BPM]   {key}{tag_str}"
            
            item = QListWidgetItem(display_text)
            item.setData(Qt.UserRole, str(file_path))
            item.setIcon(FIF.MUSIC.icon())
            self.sample_list.addItem(item)
            
    def on_sample_clicked(self, item):
        file_path = item.data(Qt.UserRole)
        if not file_path or not os.path.exists(file_path):
            self.info_label.setText("File not found")
            return
            
        self.info_label.setText(os.path.basename(file_path))
        self.waveform.load_file(file_path)
        
        url = QUrl.fromLocalFile(file_path)
        content = QMediaContent(url)
        self.player.setMedia(content)
        self.player.play()

if __name__ == "__main__":
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
    os.environ["QT_SCALE_FACTOR"] = "1"
    
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())