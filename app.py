import sys
import os
from pathlib import Path

from PyQt5.QtCore import Qt, QUrl, QMimeData, QSize, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QListWidgetItem, QListWidget, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QFileDialog, QProgressBar, QTextEdit, QMenu, QAction
)
from PyQt5.QtGui import QDrag, QIcon, QCursor
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

from qfluentwidgets import (
    FluentWindow, ListWidget, FluentIcon as FIF, SearchLineEdit, 
    PushButton, PrimaryPushButton, SubtitleLabel, Slider,
    SwitchButton, ToggleButton, TransparentToolButton, PopUpAniStackedWidget
)

import search
import indexer
import library_config
from waveform import WaveformWidget
from galaxy import GalaxyWidget
from onboarding import OnboardingView

# --- Threads ---

class SearchThread(QThread):
    results_ready = pyqtSignal(list, str)
    status_update = pyqtSignal(str)

    def __init__(self, query):
        super().__init__()
        self.query = query

    def run(self):
        try:
            # Check if it's a file path or text
            if os.path.exists(self.query):
                results = search.search_by_audio(
                    Path(self.query), 
                    str(library_config.DB_PATH), 
                    limit=50,
                    status_callback=self.status_update.emit
                )
            else:
                results = search.search_by_text(
                    self.query, 
                    str(library_config.DB_PATH), 
                    limit=50,
                    status_callback=self.status_update.emit
                )
            self.results_ready.emit(results, self.query)
        except Exception as e:
            print(f"Search error: {e}")
            self.results_ready.emit([], self.query)

class FindSimilarThread(QThread):
    results_ready = pyqtSignal(list)
    status_update = pyqtSignal(str)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        try:
            self.status_update.emit(f"Finding samples similar to {os.path.basename(self.file_path)}...")
            results = search.search_similar_to_sample(
                self.file_path,
                str(library_config.DB_PATH),
                limit=50
            )
            self.results_ready.emit(results)
        except Exception as e:
            print(f"Similarity search error: {e}")
            self.results_ready.emit([])

class IndexingThread(QThread):
    progress_signal = pyqtSignal(int, int, str)
    finished_signal = pyqtSignal(dict)

    def __init__(self, folder):
        super().__init__()
        self.folder = folder

    def run(self):
        try:
            stats = indexer.index_folder(
                Path(self.folder),
                progress_callback=self.progress_signal.emit
            )
            self.finished_signal.emit(stats)
        except Exception as e:
            print(f"Indexing error: {e}")
            self.finished_signal.emit({})

class SampleListWidget(ListWidget):
    find_similar_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_start_pos = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not (event.buttons() & Qt.LeftButton):
            return
        if (event.pos() - self.drag_start_pos).manhattanLength() < QApplication.startDragDistance():
            return

        item = self.currentItem()
        if not item:
            return

        file_path = item.data(Qt.UserRole)
        if not file_path or not os.path.exists(file_path):
            return

        drag = QDrag(self)
        mime_data = QMimeData()
        
        # Enable dragging into DAWs (Ableton, FL Studio, etc.)
        url = QUrl.fromLocalFile(file_path)
        mime_data.setUrls([url])
        drag.setMimeData(mime_data)
        
        # Set a nice drag icon if possible, or just use the emoji
        drag.exec_(Qt.CopyAction)

    def show_context_menu(self, pos):
        item = self.itemAt(pos)
        if not item:
            return

        file_path = item.data(Qt.UserRole)
        if not file_path:
            return

        menu = QMenu(self)
        
        similar_action = QAction("Find Similar Samples", self)
        similar_action.setIcon(FIF.SYNC.icon())
        similar_action.triggered.connect(lambda: self.find_similar_requested.emit(file_path))
        menu.addAction(similar_action)
        
        open_folder_action = QAction("Open in Explorer", self)
        open_folder_action.setIcon(FIF.FOLDER.icon())
        open_folder_action.triggered.connect(lambda: self.open_in_explorer(file_path))
        menu.addAction(open_folder_action)
        
        menu.exec_(self.mapToGlobal(pos))

    def open_in_explorer(self, file_path):
        folder = os.path.dirname(file_path)
        if os.path.exists(folder):
            os.startfile(folder)

class GalaxyInterface(QWidget):
    sample_selected = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.galaxy = GalaxyWidget(self)
        self.galaxy.point_clicked.connect(self.sample_selected.emit)
        layout.addWidget(self.galaxy)
        
        # Load data initially
        self.refresh_data()

    def refresh_data(self):
        try:
            db_conn = search.db.get_db(str(library_config.DB_PATH))
            table = search.db.create_samples_table(db_conn)
            self.galaxy.load_from_table(table)
        except Exception as e:
            print(f"Error loading galaxy: {e}")

    def highlight_search_results(self, results):
        paths = [r['metadata'].get('path') for r in results if 'metadata' in r]
        self.galaxy.highlight_points(paths)

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

        # Folder List
        layout.addWidget(QLabel("Indexed Folders (Right-click to Rescan/Remove):"))
        self.folder_list = ListWidget(self)
        self.folder_list.setFixedHeight(150)
        layout.addWidget(self.folder_list)
        
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

        self.load_folders()

    def load_folders(self):
        """Load indexed folders from config into the list."""
        self.folder_list.clear()
        folders = library_config.get_folders()
        for folder in folders:
            item = QListWidgetItem(folder)
            self.folder_list.addItem(item)
        
        # Add context menu for folders
        self.folder_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.folder_list.customContextMenuRequested.connect(self.show_folder_context_menu)

    def show_folder_context_menu(self, pos):
        item = self.folder_list.itemAt(pos)
        if not item:
            return
            
        folder_path = item.text()
        menu = QMenu(self)
        
        rescan_action = QAction("Rescan Folder", self)
        rescan_action.setIcon(FIF.SYNC.icon())
        rescan_action.triggered.connect(lambda: self.start_indexing(folder_path))
        menu.addAction(rescan_action)
        
        remove_action = QAction("Remove from Library", self)
        remove_action.setIcon(FIF.DELETE.icon())
        remove_action.triggered.connect(lambda: self.remove_folder(folder_path))
        menu.addAction(remove_action)
        
        menu.exec_(self.folder_list.mapToGlobal(pos))

    def remove_folder(self, folder_path):
        """Remove folder from config and database."""
        self.log_area.append(f"Removing: {folder_path}")
        try:
            # Remove from DB
            lance_db = search.db.get_db(str(library_config.DB_PATH))
            table = search.db.create_samples_table(lance_db)
            search.db.delete_folder_samples(table, folder_path)
            
            # Remove from config
            library_config.remove_folder_from_config(folder_path)
            self.load_folders()
            self.log_area.append(f"Successfully removed {folder_path}")
        except Exception as e:
            self.log_area.append(f"Error removing folder: {e}")
        
    def add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Sample Folder")
        if not folder:
            return
            
        library_config.add_folder_to_config(folder)
        self.load_folders()
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
        
        # --- State ---
        self.is_looping = False
        self.autoplay = True
        
        # --- Samples Page ---
        self.home_widget = QWidget()
        self.home_widget.setObjectName("homeWidget")
        self.home_layout = QVBoxLayout(self.home_widget)
        
        # Stacked Widget for Search vs Onboarding
        self.stacked_home = PopUpAniStackedWidget(self)
        
        # --- View 1: Main Search UI ---
        self.search_view = QWidget()
        self.search_layout = QVBoxLayout(self.search_view)
        self.search_layout.setContentsMargins(0, 0, 0, 0)

        # Search
        self.search_bar = SearchLineEdit(self)
        self.search_bar.setPlaceholderText("Search sounds (e.g. 'dark bass', '140 bpm')")
        self.search_bar.returnPressed.connect(self.perform_search)
        self.search_layout.addWidget(self.search_bar)
        
        # List
        self.sample_list = SampleListWidget(self)
        self.sample_list.itemClicked.connect(self.on_sample_clicked)
        self.sample_list.find_similar_requested.connect(self.find_similar)
        self.search_layout.addWidget(self.sample_list)
        
        # Waveform & Info
        self.info_layout = QVBoxLayout()
        self.info_label = QLabel("Select a sample to preview")
        self.info_label.setObjectName("infoLabel")
        self.info_layout.addWidget(self.info_label)
        
        self.waveform = WaveformWidget(self)
        self.waveform.seek_requested.connect(self.seek_player)
        self.info_layout.addWidget(self.waveform)

        # --- Playback Controls ---
        self.controls_layout = QHBoxLayout()
        self.controls_layout.setContentsMargins(0, 5, 0, 5)
        
        # Play/Pause
        self.play_btn = TransparentToolButton(FIF.PLAY, self)
        self.play_btn.clicked.connect(self.toggle_play)
        self.controls_layout.addWidget(self.play_btn)
        
        # Loop Toggle
        self.loop_btn = ToggleButton(FIF.SYNC, "Loop", self)
        self.loop_btn.clicked.connect(self.toggle_loop)
        self.controls_layout.addWidget(self.loop_btn)
        
        # Autoplay Toggle
        self.autoplay_btn = SwitchButton(self)
        self.autoplay_btn.setChecked(True)
        self.autoplay_btn.checkedChanged.connect(self.set_autoplay)
        self.controls_layout.addSpacing(10)
        self.controls_layout.addWidget(QLabel("Autoplay"))
        self.controls_layout.addWidget(self.autoplay_btn)
        
        self.controls_layout.addStretch(1)
        
        # Volume Slider
        self.controls_layout.addWidget(QLabel("Volume"))
        self.volume_slider = Slider(Qt.Horizontal, self)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(80)
        self.volume_slider.setFixedWidth(150)
        self.volume_slider.valueChanged.connect(self.set_volume)
        self.controls_layout.addWidget(self.volume_slider)
        
        self.info_layout.addLayout(self.controls_layout)
        self.search_layout.addLayout(self.info_layout)
        
        # --- View 2: Onboarding UI ---
        self.onboarding_view = OnboardingView(self)
        self.onboarding_view.add_folder_requested.connect(self.show_library_page)

        self.stacked_home.addWidget(self.search_view)
        self.stacked_home.addWidget(self.onboarding_view)
        
        self.home_layout.addWidget(self.stacked_home)
        
        # --- Library Page ---
        self.library_interface = LibraryInterface(self)

        # --- Galaxy Page ---
        self.galaxy_interface = GalaxyInterface(self)
        self.galaxy_interface.sample_selected.connect(self.on_galaxy_sample_selected)

        # --- Navigation ---
        self.addSubInterface(self.home_widget, icon=FIF.MUSIC, text='Samples')
        self.addSubInterface(self.galaxy_interface, icon=FIF.VIEW, text='Galaxy')
        self.addSubInterface(self.library_interface, icon=FIF.FOLDER, text='Library')
        
        # Media Player
        self.player = QMediaPlayer()
        self.player.setVolume(80)
        self.player.positionChanged.connect(self.update_playhead)
        self.player.stateChanged.connect(self.on_player_state_changed)
        self.player.mediaStatusChanged.connect(self.on_media_status_changed)

        # Initial check for empty library
        self.check_library_empty()

    def check_library_empty(self):
        """Show onboarding if library is empty, otherwise show search."""
        try:
            db = search.db.get_db(str(library_config.DB_PATH))
            table = search.db.create_samples_table(db)
            count = search.db.count_samples(table)
            if count == 0:
                self.stacked_home.setCurrentWidget(self.onboarding_view)
            else:
                self.stacked_home.setCurrentWidget(self.search_view)
        except Exception as e:
            print(f"Error checking library status: {e}")
            self.stacked_home.setCurrentWidget(self.onboarding_view)

    def show_library_page(self):
        """Switch to the library management tab."""
        self.switchTo(self.library_interface)

    def toggle_play(self):
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def on_player_state_changed(self, state):
        if state == QMediaPlayer.PlayingState:
            self.play_btn.setIcon(FIF.PAUSE)
        else:
            self.play_btn.setIcon(FIF.PLAY)

    def toggle_loop(self):
        self.is_looping = self.loop_btn.isChecked()

    def set_autoplay(self, checked):
        self.autoplay = checked

    def set_volume(self, value):
        self.player.setVolume(value)

    def seek_player(self, percentage):
        if self.player.duration() > 0:
            pos = int(percentage * self.player.duration())
            self.player.setPosition(pos)

    def update_playhead(self, position):
        duration = self.player.duration()
        if duration > 0:
            progress = position / duration
            self.waveform.set_progress(progress)

    def on_media_status_changed(self, status):
        if status == QMediaPlayer.EndOfMedia and self.is_looping:
            self.player.setPosition(0)
            self.player.play()

    def perform_search(self):
        query = self.search_bar.text().strip()
        if not query:
            return

        # Don't search if it's showing "Similar to: ..." from a find_similar action
        if query.startswith("Similar to:"):
            self.search_bar.clear()
            return

        self.search_bar.setEnabled(False)
        self.info_label.setText("Searching...")

        self.search_thread = SearchThread(query)
        self.search_thread.results_ready.connect(self.handle_results)
        self.search_thread.status_update.connect(self.update_status)
        self.search_thread.finished.connect(lambda: self.search_bar.setEnabled(True))
        self.search_thread.start()

    def handle_results(self, results, query_info: str = None):
        self.sample_list.clear()

        if not results:
            item = QListWidgetItem("No results found.")
            item.setFlags(Qt.NoItemFlags)
            self.sample_list.addItem(item)
            self.galaxy_interface.galaxy.clear_highlights()
            return

        # Tag-to-Emoji mapping for quick visual scanning
        tag_icons = {
            "Kick": "🥁", "Snare": "🥁", "Hi-Hat": "📀", "Bass": "🎸", "808": "🔊",
            "Synth": "🎹", "Vocals": "🎤", "FX": "✨", "Ambient": "☁️", "Loop": "🔁",
            "One Shot": "🎯", "Trap": "🔥", "House": "🏠", "Techno": "🤖"
        }

        for r in results:
            meta = r.get('metadata', {})
            filename = meta.get('filename', 'Unknown')
            file_path = meta.get('path', '')
            bpm = meta.get('bpm', 0)
            key = meta.get('key', '')
            tags = meta.get('tags', [])
            distance = r.get('_distance', 0)

            # Pick best icon based on top tag
            icon_emoji = "🎵"
            if tags:
                for tag in tags:
                    if tag in tag_icons:
                        icon_emoji = tag_icons[tag]
                        break

            # Format tags
            tag_str = ""
            if tags:
                tag_str = " | ".join(tags[:3])
                tag_str = f"   ({tag_str})"

            # Show distance for similarity searches
            dist_str = f"  [{distance:.3f}]" if distance else ""
            display_text = f"{icon_emoji}  {filename}   [{bpm:.1f} BPM]   {key}{tag_str}{dist_str}"

            item = QListWidgetItem(display_text)
            item.setData(Qt.UserRole, str(file_path))
            
            # We can also set a tooltip with more info
            item.setToolTip(f"Path: {file_path}\nTags: {', '.join(tags)}")
            
            self.sample_list.addItem(item)

        # Highlight search results in galaxy view
        self.galaxy_interface.highlight_search_results(results)
        self.info_label.setText(f"Found {len(results)} samples")

    def find_similar(self, file_path: str):
        """Find samples similar to the given file."""
        self.search_bar.setEnabled(False)
        self.search_bar.setText(f"Similar to: {Path(file_path).name}")

        self.find_similar_thread = FindSimilarThread(file_path)
        self.find_similar_thread.results_ready.connect(self.handle_results)
        self.find_similar_thread.status_update.connect(self.update_status)
        self.find_similar_thread.finished.connect(lambda: self.search_bar.setEnabled(True))
        self.find_similar_thread.start()

    def update_status(self, message: str):
        """Update status display (for loading messages etc.)."""
        self.info_label.setText(message)

    def on_galaxy_sample_selected(self, meta: dict):
        """Handle sample selection from galaxy view."""
        file_path = meta.get("path", "")
        if not file_path or not os.path.exists(file_path):
            self.info_label.setText("File not found")
            return

        # Switch to samples tab
        self.switchTo(self.home_widget)

        # Update info and waveform
        self.info_label.setText(os.path.basename(file_path))
        self.waveform.load_file(file_path)

        # Play the sample
        url = QUrl.fromLocalFile(file_path)
        content = QMediaContent(url)
        self.player.setMedia(content)
        self.player.play()
            
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