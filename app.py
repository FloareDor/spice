import sys
import os
from pathlib import Path

from PyQt5.QtCore import Qt, QUrl, QMimeData, QSize
from PyQt5.QtWidgets import QApplication, QListWidgetItem, QListWidget
from PyQt5.QtGui import QDrag, QIcon

from qfluentwidgets import FluentWindow, ListWidget, FluentIcon as FIF

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
        
        # Create the sample list
        self.sample_list = SampleListWidget(self)
        
        # Add some dummy items (real files for testing drag-and-drop)
        # Using paths found in the project
        files = [
            r"E:\cmu-e\spice\test_pack\MOONBOY - Chaos Ableton Project File Project\Samples\Imported\Break It Down... 1.wav",
            r"E:\cmu-e\spice\test_pack\MOONBOY - Chaos Ableton Project File Project\Samples\Imported\MOONBOY - Chaos Impact 1.wav",
            r"E:\cmu-e\spice\test_pack\MOONBOY - Chaos Ableton Project File Project\Samples\Imported\Siren-1.wav",
            r"E:\cmu-e\spice\test_pack\MOONBOY - Chaos Ableton Project File Project\Samples\Imported\Djent Riff (Loop in E).wav",
            r"E:\cmu-e\spice\test_pack\MOONBOY - Chaos Ableton Project File Project\Samples\Imported\Feedback - Changing & Whammy Down.wav"
        ]
        
        for f in files:
            path = Path(f)
            if path.exists():
                item = QListWidgetItem(path.name)
                item.setData(Qt.UserRole, str(path))
                # Add a music icon to each item
                item.setIcon(FIF.MUSIC.icon())
                self.sample_list.addItem(item)
            else:
                print(f"Warning: File not found {f}")

        # Add to sub interface
        self.addSubInterface(self.sample_list, icon=FIF.MUSIC, text='Samples')
        
        # Select the first item by default
        if self.sample_list.count() > 0:
            self.sample_list.setCurrentRow(0)

if __name__ == "__main__":
    # Ensure high DPI scaling works correctly
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
    os.environ["QT_SCALE_FACTOR"] = "1"
    
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
