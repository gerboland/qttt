from PySide2 import QtCore
from PySide2.QtCore import QObject, QMutex, QWaitCondition


class QMLRenderer(QObject):
    boardChanged = QtCore.Signal()
    playerChanged = QtCore.Signal(int)
    gameOver = QtCore.Signal(str)
    playAgain = QtCore.Signal()

    def __init__(self, parent=None) -> None:
        super(QMLRenderer, self).__init__(parent)
        self.mutex = QMutex()
        self.wait = QWaitCondition()
        self.playAgain.connect(self.onPlayAgain)

    def render(self, board) -> None:
        # board change
        self.boardChanged.emit()

    def prompt_player(self, player: int) -> None:
        # signal game to ask for user input?
        self.playerChanged.emit(player)

    def game_over(self, player, reward) -> None:
        self.gameOver.emit(player)

        # Block until user chooses to play again
        self.mutex.lock()
        self.wait.wait(self.mutex)
        self.mutex.unlock()

    def onPlayAgain(self):
        self.mutex.lock()
        self.wait.wakeAll()
        self.mutex.unlock()
