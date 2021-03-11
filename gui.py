#
# A GUI for ps-auxw's CLI-P (commandline driven semantic image search using OpenAI's CLIP)
#

import sys
from io import StringIO
import contextlib

from PyQt5.QtCore import (
    QTimer,
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout,
    QLabel, QLineEdit, QTextEdit,
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
        self.searchOutput = QTextEdit()
        self.searchOutput.setReadOnly(True)
        self.searchInput = QLineEdit()
        self.searchInput.returnPressed.connect(self.handleSearchInput)

        vBox.addWidget(self.cvLabel)
        vBox.addWidget(self.searchOutput)
        vBox.addWidget(self.searchInput)

        self.setCentralWidget(widget)

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
        QTimer.singleShot(0, self.delayLoadModules)  # (Run after all events.)

    def delayLoadModules(self):
        QTimer.singleShot(50, self.loadModules)  # (Delay a bit further in the hopes it might actually work.)

    def appendSearchOutput(self, lines):
        # Skip calls with nothing to convey.
        if lines == None or lines == "":
            return
        # Strip last newline, but only that.
        # That way, an empty line at the end
        # can be requested by "...\n\n".
        # Otherwise, we could simply use: lines.rstrip('\n')
        if lines[-1] == '\n':
            lines = lines[:-1]
        self.searchOutput.append(lines)

    def handleSearchInput(self):
        inputText = self.searchInput.text()

        search = self.search
        if search == None:
            self.appendSearchOutput("Search not ready, yet...")
            return
        search.in_text = inputText.strip()

        f = StringIO()
        iteration_done = None
        with contextlib.redirect_stdout(f):
            iteration_done = search.do_command()
        self.appendSearchOutput(f.getvalue())
        del f
        if iteration_done:
            return

        f = StringIO()
        with contextlib.redirect_stdout(f):
            search.do_search()
        self.appendSearchOutput(f.getvalue())
        del f

        f = StringIO()
        with contextlib.redirect_stdout(f):
            search.do_display()
        self.appendSearchOutput(f.getvalue())
        del f


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
