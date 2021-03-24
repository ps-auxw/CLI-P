#
# A GUI for ps-auxw's CLI-P (commandline driven semantic image search using OpenAI's CLIP)
#

import sys
import time
from io import StringIO
import contextlib

from PyQt5.QtCore import (
    pyqtSignal,
    Qt,
    QItemSelectionModel,
    QTimer,
)
from PyQt5.QtGui import (
    QStandardItemModel, QStandardItem,
    QImage, QPixmap,
)
from PyQt5.QtWidgets import (
    qApp,
    QApplication, QMainWindow, QWidget,
    QSizePolicy,
    QHBoxLayout, QVBoxLayout, QScrollBar, QTabWidget, QToolBar,
    QComboBox, QLabel, QPushButton, QTextEdit,
    QTableView,
)

# Load delayed, so the GUI is already visible,
# as this may take a long time.
query_index = None

class HistoryComboBox(QComboBox):
    pageChangeRequested = pyqtSignal(bool)

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
        # On PageUp/Down, emit a signal
        # (so this can control, e.g., scrolling of the console log).
        elif key == Qt.Key_PageUp or key == Qt.Key_PageDown:
            self.pageChangeRequested.emit(key == Qt.Key_PageUp)
            # Don't prevent default handling of the key press.
            super(HistoryComboBox, self).keyPressEvent(ev)
        # Otherwise, propagate key press further.
        else:
            super(HistoryComboBox, self).keyPressEvent(ev)

class MainWindow(QMainWindow):
    class OurTabPage(QWidget):
        resized = pyqtSignal()
        def __init__(self, parent=None):
            super(MainWindow.OurTabPage, self).__init__(parent)
        def resizeEvent(self, ev):
            super(MainWindow.OurTabPage, self).resizeEvent(ev)
            self.resized.emit()

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.search = None
        self.searchResultSelected = None

        # TODO: Take from config db.
        self.resize(1600, 900)

        centralWidget = QWidget()
        centralVBox = QVBoxLayout(centralWidget)

        self.tabWidget = QTabWidget()


        # Page 1: Console
        self.consoleTabPage = QWidget()
        consoleVBox = QVBoxLayout(self.consoleTabPage)

        self.infoLabel = QLabel(
            "ps-auxw says, \"CLI-P is commandline driven semantic image search using OpenAI's CLIP\"\n"
            "canvon says, \"This is a GUI for ps-auxw's CLI-P\"")
        self.searchOutput = QTextEdit()
        self.searchOutput.setReadOnly(True)

        consoleVBox.addWidget(self.infoLabel)
        consoleVBox.addWidget(self.searchOutput)
        self.tabWidget.addTab(self.consoleTabPage, "&1 Console")


        # Page 2: Images
        self.imagesTabPage = self.OurTabPage()
        self.imagesTabPage.resized.connect(self.imagesTabPageResized)
        imagesVBox = QVBoxLayout(self.imagesTabPage)

        imgsToolBar = QToolBar()
        self.imagesToolBar = imgsToolBar
        self.imagesActionAddTag = imgsToolBar.addAction("Add to tag (&+)", self.imagesActionAddTagTriggered)
        self.imagesActionDelTag = imgsToolBar.addAction("Del from tag (&-)", self.imagesActionDelTagTriggered)
        self.imagesActionAddTag.setShortcut("Ctrl+T")
        self.imagesActionDelTag.setShortcut("Ctrl+Shift+T")

        self.imageLabel = QLabel()
        self.imagesTableView = QTableView()
        self.imagesTableView.setEditTriggers(QTableView.NoEditTriggers)
        self.imagesTableView.activated.connect(self.searchResultsActivated)

        imagesVBox.addWidget(self.imagesToolBar)
        imagesVBox.addWidget(self.imageLabel)
        imagesVBox.addWidget(self.imagesTableView)
        self.tabWidget.addTab(self.imagesTabPage, "&2 Images")


        self.searchHint = QLabel()


        # Search input box & go button
        inputHBox = QHBoxLayout()

        self.searchInputLabel = QLabel("&Search")
        self.searchInputLabel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.searchInput = HistoryComboBox()
        self.searchInput.setEditable(True)
        self.searchInput.pageChangeRequested.connect(self.searchInputPageChangeRequested)
        #
        # This fired too often, but we only want to search
        # when the user finally hits return...
        #self.searchInput.activated.connect(self.handleSearchInput)
        #
        self.searchInputLabel.setBuddy(self.searchInput)

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

        inputHBox.addWidget(self.searchInputLabel)
        inputHBox.addWidget(self.searchInput)
        inputHBox.addWidget(self.searchInputButton)


        centralVBox.addWidget(self.tabWidget)
        centralVBox.addWidget(self.searchHint)
        centralVBox.addLayout(inputHBox)

        self.setCentralWidget(centralWidget)
        self.searchInput.setFocus()

        self.createSearchResultsModel()

    def imagesTabPageResized(self):
        contents = self.imagesTabPage.contentsRect()
        self.imageLabel.setMaximumSize(contents.width(), contents.height() * 8 / 10)  # 80%
        self.imageLabel.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)

    def searchInputPageChangeRequested(self, pageUp):
        w = self.tabWidget.currentWidget()
        if w is self.consoleTabPage:
            # Scroll console log.
            out = self.searchOutput
            height = out.viewport().height()
            # This didn't work:
            #out.scrollContentsBy(0, height * (-1 if pageUp else 1))
            # This worked, but doesn't provide overlap:
            #out.verticalScrollBar().triggerAction(QScrollBar.SliderPageStepSub if pageUp else QScrollBar.SliderPageStepAdd)
            # So we end up with this:
            # (It scrolls by half the viewport height.)
            vbar = out.verticalScrollBar()
            vbar.setValue(vbar.value() + height * (-1 if pageUp else 1) / 2)
        elif w is self.imagesTabPage:
            # Scroll in & activate search results.
            view = self.imagesTableView
            model = view.model()
            selectionModel = view.selectionModel()
            index = selectionModel.currentIndex()
            nextIndex = None
            if not index.isValid():
                nextIndex = model.index(0, 0)
            else:
                nextIndex = index.siblingAtRow(index.row() + (-1 if pageUp else 1))
            if not nextIndex.isValid():
                return
            selectionModel.setCurrentIndex(nextIndex, QItemSelectionModel.SelectCurrent)
            self.searchResultsActivated(nextIndex)

    def loadModules(self):
        global query_index
        if query_index == None:
            self.appendSearchOutput("Loading query-index...")
            qApp.processEvents()
            loadStart = time.perf_counter()
            query_index = __import__('query-index')  # TODO: Adjust file name.
            loadTime = time.perf_counter() - loadStart
            self.appendSearchOutput(f"Loaded query-index: {loadTime:.4f}s")
            qApp.processEvents()
        if self.search == None:
            self.appendSearchOutput("Instantiating search...")
            qApp.processEvents()
            instantiateStart = time.perf_counter()
            self.search = query_index.Search()
            instantiateTime = time.perf_counter() - instantiateStart
            self.appendSearchOutput(f"Instantiated search: {instantiateTime:.4f}s")
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
        self.clearSearchResultsModel()

        iteration_done = self.stdoutSearchOutput(search.do_command)
        if iteration_done:
            # Check for q (quit) command.
            if search.running_cli == False:
                self.close()
            return

        self.stdoutSearchOutput(search.do_search)
        n_results = 0 if search.results is None else len(search.results)
        if not n_results > 0:
            self.appendSearchOutput("No results.")
            return
        #self.stdoutSearchOutput(search.do_display)
        self.appendSearchOutput(f"Building results model for {n_results} results...")
        for result in search.prepare_results():
            self.appendToSearchResultsModel(result)
        self.appendSearchOutput(f"Built results model with {self.searchResultsModel.rowCount()} entries.")

        # Already activate Images tab and load first image...
        self.tabWidget.setCurrentWidget(self.imagesTabPage)
        view = self.imagesTableView
        nextIndex = view.model().index(0, 0)
        if nextIndex.isValid():
            view.selectionModel().setCurrentIndex(nextIndex, QItemSelectionModel.SelectCurrent)
            self.searchResultsActivated(nextIndex)

    def createSearchResultsModel(self):
        model = QStandardItemModel(0, 4)
        model.setHorizontalHeaderLabels(["score", "fix_idx", "face_id", "filename"])
        self.searchResultsModel = model
        self.imagesTableView.setModel(model)
        self.imagesTableView.horizontalHeader().setStretchLastSection(True)

    def clearSearchResultsModel(self):
        self.searchResultSelected = None
        #
        # Don't use clear(), as that will get rid of the header labels
        # and column count, too...
        #self.searchResultsModel.clear()
        self.searchResultsModel.setRowCount(0)
        #
        self.imageLabel.clear()

    def prepareSearchResultsModelEntry(self, result):
        scoreItem  = QStandardItem(str(result.score))
        fixIdxItem = QStandardItem(str(result.fix_idx))
        faceIdItem = QStandardItem(str(result.face_id))
        tfnItem    = QStandardItem(str(result.tfn))
        items = [scoreItem, fixIdxItem, faceIdItem, tfnItem]
        for item in items:
            item.setData(result)
        return items

    def appendToSearchResultsModel(self, result):
        model = self.searchResultsModel
        items = self.prepareSearchResultsModelEntry(result)
        result.gui_rowOffset = model.rowCount()
        model.appendRow(items)

    def recreateSearchResultsModelRow(self, result):
        search = self.search
        if search is None:
            self.appendSearchOutput("Search instance missing.")
            return
        rowOffset = result.gui_rowOffset
        # Recreate Search.Result instance.
        # (e.g., rereads annotations/tags.)
        search.tried_j = -1
        search.last_vector = None
        recreatedResult, j, _ = search.prepare_result(result.results_j)
        if j is None:
            self.appendSearchOutput(f"Failed to recreate search results model row {rowOffset+1}: Prepare result indicated end of results.")
            return
        elif recreatedResult is None:
            self.appendSearchOutput(f"Failed to recreate search results model row {rowOffset+1}: Prepare result indicated skip.")
            return
        recreatedResult.gui_rowOffset = rowOffset
        # Update Qt-side model.
        model = self.searchResultsModel
        items = self.prepareSearchResultsModelEntry(recreatedResult)
        for columnOffset in range(model.columnCount()):
            model.setItem(rowOffset, columnOffset, items[columnOffset])
        return recreatedResult

    def searchResultsActivated(self, index):
        result = index.data(Qt.UserRole + 1)
        self.showSearchResult(result, force=True)

    def showSearchResult(self, result, force=False):
        if not force and self.searchResultSelected is result:
            return
        self.searchResultSelected = result
        if result is None:
            return
        self.appendSearchOutput(result.format_output())
        # Prepare image.
        try:
            size = self.imageLabel.maximumSize()
            max_res = (size.width(), size.height())
            image = self.search.prepare_image(result, max_res)
            if image is None:
                raise RuntimeError("No image.")
        except Exception as ex:
            self.appendSearchOutput(f"Error preparing image: {ex}")
            return
        # Convert prepared image to Qt/GUI.
        qtImage = QImage(image.data, image.shape[1], image.shape[0], 3 * image.shape[1], QImage.Format_RGB888).rgbSwapped()
        self.imageLabel.setPixmap(QPixmap.fromImage(qtImage))

    def updateSearchResultSelected(self, updateCode):
        result = self.searchResultSelected
        if result is None:
            self.appendSearchOutput("Update search result selected: No search result selected.")
            return
        self.stdoutSearchOutput(lambda: updateCode(result))
        recreatedResult = self.recreateSearchResultsModelRow(result)
        if recreatedResult is None:
            return
        self.showSearchResult(recreatedResult, force=True)

    def imagesActionAddTagTriggered(self):
        search = self.search
        if search is None:
            self.appendSearchOutput("Search instance missing.")
            return
        self.updateSearchResultSelected(search.maybe_add_tag)

    def imagesActionDelTagTriggered(self):
        search = self.search
        if search is None:
            self.appendSearchOutput("Search instance missing.")
            return
        self.updateSearchResultSelected(search.maybe_del_tag)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setApplicationName("CLI-P GUI")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
