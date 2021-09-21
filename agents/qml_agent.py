import sys
from PySide2 import QtCore
from PySide2.QtCore import QObject, QMutex, QWaitCondition


class QMLAgent(QObject):
    type = "qml"
    userTileSelected = QtCore.Signal(str, int)

    def __init__(self, player, env=None, parent=None):
        super(QMLAgent, self).__init__(parent)
        self.player = player
        self.mutex = QMutex()
        self.wait = QWaitCondition()
        self.choice = None

        self.userTileSelected.connect(self.onUserTileSelected)

    def act(self, state, available_actions):
        while True:
            # Block until user clicks a tile in QML
            self.mutex.lock()
            self.wait.wait(self.mutex)
            action = int(self.choice)
            self.mutex.unlock()

            if action not in available_actions:
                print(f"Move {self.choice} is already taken, please try another")
                continue

            return action

    @QtCore.Slot(str, int)
    def onUserTileSelected(self, player: str, index: int):
        if player == self.player:
            self.mutex.lock()
            self.wait.wakeAll()
            self.choice = index
            self.mutex.unlock()
