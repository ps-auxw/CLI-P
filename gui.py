#
# A GUI for ps-auxw's CLI-P (commandline driven semantic image search using OpenAI's CLIP)
#

import sys
from PyQt5.QtCore import (
    pyqtSlot,
    QTimer,
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout,
    QLabel, QLineEdit,
)

# Load delayed, so the GUI is already visible,
# as this may take a long time.
query_index = None

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.search = None

        # TODO: Take from config db.
        self.resize(1600, 900)

        widget = QWidget(self)
        vBox = QVBoxLayout(widget)

        self.cvLabel = QLabel()
        self.searchInput = QLineEdit()

        vBox.addWidget(self.cvLabel)
        vBox.addWidget(self.searchInput)

        self.setCentralWidget(widget)

    #@pyqtSlot()
    def loadModules(self):
        global query_index
        if query_index == None:
            # TODO: Convey this information via the GUI.
            print("Loading query-index...")
            query_index = __import__('query-index')  # TODO: Adjust file name.
            print("Loaded query-index.")
        if self.search == None:
            print("Instantiating search...")
            self.search = query_index.Search()
            print("Instantiated search.")

    def showEvent(self, ev):
        super(MainWindow, self).showEvent(ev)
        if ev.spontaneous():
            print("Spontaneous show event.")
            self.loadModules()
        else:
            print("Non-spontaneous show event. Delaying via QTimer::singleShot(0, ...)...")
            QTimer.singleShot(0, self.loadModules)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
