import sys
import os
from pathlib import Path

from PyQt5.QtCore import Qt, QUrl, QMimeData, QSize, QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QListWidgetItem, QListWidget, QWidget, QVBoxLayout
from PyQt5.QtGui import QDrag, QIcon

from qfluentwidgets import FluentWindow, ListWidget, FluentIcon as FIF, SearchLineEdit

import search

class SearchThread(QThread):
    results_ready = pyqtSignal(list)

    def __init__(self, query_text):
        super().__init__()
        self.query_text = query_text

    def run(self):
        try:
            # Search for "query_text" in the local database
            # We use a limit of 50 for now
            results = search.search_by_text(self.query_text, "./localvibe.lance", limit=50)
            self.results_ready.emit(results)
        except Exception as e:
            print(f"Search thread error: {e}")
            self.results_ready.emit([])

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
        
        # Get the file path stored in the item's data
        file_path = item.data(Qt.UserRole)
        if not file_path:
            return

        # Create MIME data
        mime = QMimeData()
        url = QUrl.fromLocalFile(file_path)
        mime.setUrls([url])
        
        # Create drag object
        drag = QDrag(self)
        drag.setMimeData(mime)
        
        # Execute drag
        drag.exec_(Qt.CopyAction)

class MainWindow(FluentWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LocalVibe")
        self.resize(900, 700)
        
        # Center window on screen
        desktop = QApplication.primaryScreen().availableGeometry()
        w, h = desktop.width(), desktop.height()
        self.move(w//2 - self.width()//2, h//2 - self.height()//2)
        
        # Create a central container widget for the 'Samples' page
        self.home_widget = QWidget()
        self.home_widget.setObjectName("homeWidget")
        self.home_layout = QVBoxLayout(self.home_widget)
        
        # Search Bar
        self.search_bar = SearchLineEdit(self)
        self.search_bar.setPlaceholderText("Search sounds (e.g. 'dark bass', '140 bpm')")
        self.search_bar.returnPressed.connect(self.perform_search)
        self.home_layout.addWidget(self.search_bar)
        
        # Sample List
        self.sample_list = SampleListWidget(self)
        self.home_layout.addWidget(self.sample_list)
        
        # Add to sub interface
        self.addSubInterface(self.home_widget, icon=FIF.MUSIC, text='Samples')
        
        # Populate with initial data (List all samples)
        self.load_initial_samples()

    def load_initial_samples(self):
        """List all samples on startup"""
        try:
            # We can reuse search.list_samples logic or just run a wildcard search
            # For now, let's just trigger an empty search or a default query
            # Or better, just list the files we know about from the previous step
            # for a fast startup, but let's try to query the DB if possible.
            # search.py doesn't expose a clean 'get_all' without print, 
            # so let's just leave it empty or show the dummy files for now
            # until user searches. 
            pass 
        except Exception as e:
            print(f"Error loading initial samples: {e}")

    def perform_search(self):
        query = self.search_bar.text().strip()
        if not query:
            return
            
        self.search_bar.setEnabled(False) # Disable while searching to prevent double submit
        
        self.search_thread = SearchThread(query)
        self.search_thread.results_ready.connect(self.handle_results)
        self.search_thread.finished.connect(lambda: self.search_bar.setEnabled(True))
        self.search_thread.start()

    def handle_results(self, results):
        self.sample_list.clear()
        
        if not results:
            item = QListWidgetItem("No results found.")
            item.setFlags(Qt.NoItemFlags) # Make unselectable
            self.sample_list.addItem(item)
            return

        for r in results:
            meta = r.get('metadata', {})
            filename = meta.get('filename', 'Unknown')
            file_path = meta.get('path', '')
            bpm = meta.get('bpm', 0)
            key = meta.get('key', '')
            
            # Format display text: "Filename (BPM - Key)"
            display_text = f"{filename}   [{bpm:.1f} BPM]   {key}"
            
            item = QListWidgetItem(display_text)
            item.setData(Qt.UserRole, str(file_path))
            item.setIcon(FIF.MUSIC.icon())
            self.sample_list.addItem(item)

if __name__ == "__main__":
    # Ensure high DPI scaling works correctly
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
    os.environ["QT_SCALE_FACTOR"] = "1"
    
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
