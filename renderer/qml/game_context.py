from enum import Enum
from typing import Tuple, List, Optional
from PySide2 import QtCore

from renderer.qml.qml_board import QMLBoard
from envs.classic_env import ClassicTicTacToeEnv
from renderer.qml_renderer import QMLRenderer


class QMLGameContext(QtCore.QObject):
    def __init__(
        self, env: ClassicTicTacToeEnv, agents, renderer: QMLRenderer, parent=None
    ):
        super(QMLGameContext, self).__init__(parent)
        self._env = env  # NOTE: belongs to another thread
        self._agents = agents
        self._board = QMLBoard(self._env.board)
        self._gameOver: bool = False
        self._renderer = renderer

        renderer.playerChanged.connect(self.onPlayerChanged)
        renderer.gameOver.connect(self.onGameOver)

    # QProperties
    _activePlayerChanged = QtCore.Signal()
    _scoresChanged = QtCore.Signal()
    gameOver = QtCore.Signal(str)
    # Fired when game is reset
    resetEmitted = QtCore.Signal()

    def getBoard(self) -> QMLBoard:
        return self._board

    @QtCore.Property(str, notify=_activePlayerChanged)
    def activePlayer(self) -> str:
        return self._env.current_player()

    @QtCore.Property("QVariantList", notify=_scoresChanged)
    def scores(self) -> Tuple[int, int]:
        return tuple(1, 1)

    # Workaround for Pyside2 bug in Qt5.15: https://bugreports.qt.io/browse/PYSIDE-1426
    board = QtCore.Property(QtCore.QObject, getBoard, constant=True)

    @QtCore.Slot(int)
    def onPlayerChanged(self, player: str) -> None:
        self._activePlayerChanged.emit()

    @QtCore.Slot(int)
    def onGameOver(self, winner: str) -> None:
        self.gameOver.emit(winner)

    # # Called by user to reset the game.
    @QtCore.Slot()
    def newGame(self):
        self._activePlayerChanged.emit()
        self.resetEmitted.emit()

    @QtCore.Slot(int)
    def doMove(self, index: int) -> bool:
        player = self.activePlayer
        if player == "X":
            self._agents[0].userTileSelected.emit(player, index)
        else:
            self._agents[1].userTileSelected.emit(player, index)

    @QtCore.Slot()
    def playAgain(self):
        self._renderer.playAgain.emit()

    # Called by user to reset the game.
    @QtCore.Slot()
    def resetGame(self) -> None:
        self.env.board.reset()
