#
# A GUI for ps-auxw's CLI-P (commandline driven semantic image search using OpenAI's CLIP)
#

import sys
from io import StringIO
import contextlib

from PyQt5.QtCore import (
    Qt,
    QTimer,
)
from PyQt5.QtWidgets import (
    qApp,
    QApplication, QMainWindow, QWidget,
    QSizePolicy,
    QHBoxLayout, QVBoxLayout,
    QComboBox, QLabel, QPushButton, QTextEdit,
)

# Load delayed, so the GUI is already visible,
# as this may take a long time.
query_index = None

class HistoryComboBox(QComboBox):
    def __init__(self, parent=None):
        super(HistoryComboBox, self).__init__(parent)
        self.isShowingPopup = False
        self._defaultButton = None

    def defaultButton(self):
        return self._defaultButton

    def setDefaultButton(self, button):
        self._defaultButton = button

    def showPopup(self):
        super(HistoryComboBox, self).showPopup()
        self.isShowingPopup = True

    def hidePopup(self):
        self.isShowingPopup = False
        super(HistoryComboBox, self).hidePopup()

    def keyPressEvent(self, ev):
        key = ev.key()
        # On Return (Here, Enter is located on the key pad, instead!),
        # activate the associated default button.
        # Necessary in non-dialogs.
        if key == Qt.Key_Return and self._defaultButton != None:
            # Propagate further, first.
            # Necessary so the user input still gets added to the list.
            super(HistoryComboBox, self).keyPressEvent(ev)
            # Then, activate the default button.
            self._defaultButton.click()
        # On up/down, ensure the popup opens.
        elif (key == Qt.Key_Up or key == Qt.Key_Down) and not self.isShowingPopup:
            self.showPopup()
            # Don't prevent default handling of the key press.
            super(HistoryComboBox, self).keyPressEvent(ev)
        # Otherwise, propagate key press further.
        else:
            super(HistoryComboBox, self).keyPressEvent(ev)

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.search = None

        self.setWindowTitle("CLI-P GUI")
        # TODO: Take from config db.
        self.resize(1600, 900)

        widget = QWidget(self)
        vBox = QVBoxLayout(widget)

        self.infoLabel = QLabel(
            "ps-auxw says, \"CLI-P is commandline driven semantic image search using OpenAI's CLIP\"\n"
            "canvon says, \"This is a GUI for ps-auxw's CLI-P\"")
        self.cvLabel = QLabel()
        self.searchOutput = QTextEdit()
        self.searchOutput.setReadOnly(True)
        self.searchHint = QLabel()


        inputHBox = QHBoxLayout()

        self.searchInput = HistoryComboBox()
        self.searchInput.setEditable(True)
        # This fired too often, but we only want to search
        # when the user finally hits return...
        #self.searchInput.activated.connect(self.handleSearchInput)

        self.searchInputButton = QPushButton()
        #
        # Doesn't work without a Dialog:
        #self.searchInputButton.setAutoDefault(True)
        #self.searchInputButton.setDefault(True)
        # ..so, do this instead:
        self.searchInput.setDefaultButton(self.searchInputButton)
        #
        self.searchInputButton.setText("&Go")
        self.searchInputButton.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.searchInputButton.clicked.connect(self.handleSearchInput)

        inputHBox.addWidget(self.searchInput)
        inputHBox.addWidget(self.searchInputButton)


        vBox.addWidget(self.infoLabel)
        vBox.addWidget(self.cvLabel)
        vBox.addWidget(self.searchOutput)
        vBox.addWidget(self.searchHint)
        vBox.addLayout(inputHBox)

        self.setCentralWidget(widget)
        self.searchInput.setFocus()

    def loadModules(self):
        global query_index
        if query_index == None:
            self.appendSearchOutput("Loading query-index...")
            qApp.processEvents()
            query_index = __import__('query-index')  # TODO: Adjust file name.
            self.appendSearchOutput("Loaded query-index.")
            qApp.processEvents()
        if self.search == None:
            self.appendSearchOutput("Instantiating search...")
            qApp.processEvents()
            self.search = query_index.Search()
            self.appendSearchOutput("Instantiated search.")
            qApp.processEvents()

            self.appendSearchOutput("\n" + self.search.init_msg)
            self.searchHint.setText("Short help: " + self.search.prompt_prefix)

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

    def stdoutSearchOutput(self, code):
        ret = None
        with contextlib.closing(StringIO()) as f:
            with contextlib.redirect_stdout(f):
                ret = code()
            self.appendSearchOutput(f.getvalue())
        return ret

    def handleSearchInput(self):
        inputText  = self.searchInput.currentText()
        inputIndex = self.searchInput.currentIndex()
        storedText = None if inputIndex == -1 else self.searchInput.itemText(inputIndex)

        search = self.search
        if search == None:
            self.appendSearchOutput("Search not ready, yet...")
            return
        self.appendSearchOutput(">>> " + inputText)
        search.in_text = inputText.strip()
        if storedText != inputText:
            self.searchInput.addItem(inputText)
        self.searchInput.clearEditText()

        iteration_done = self.stdoutSearchOutput(search.do_command)
        if iteration_done:
            # Check for q (quit) command.
            if search.running_cli == False:
                self.close()
            return

        self.stdoutSearchOutput(search.do_search)
        self.stdoutSearchOutput(search.do_display)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
