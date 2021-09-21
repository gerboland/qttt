import typing
import PySide2.QtCore as QtCore
from renderer.qml.qml_tile import QMLTile
from board_interface import BoardInterface
from tile_interface import TileInterface


def map_to_mark(player: int) -> str:
    if player == 1:
        return "X"
    elif player == 2:
        return "O"
    else:
        return " "


class QMLBoard(QtCore.QAbstractTableModel):
    """
    Is a copy of the Board that lives in the Qt Thread, updated from the original
    by cross-thread signal/slot.
    """

    TileRole = QtCore.Qt.UserRole + 1000

    def __init__(self, board: BoardInterface, parent=None):
        super(QMLBoard, self).__init__(parent)
        self.board_size = board.board_size
        self._doReset()

        board.board_reset.connect(self.reset)
        board.tile_changed.connect(self.onTileChanged)

    @QtCore.Slot(int, TileInterface)
    def onTileChanged(self, index: int, tile: TileInterface):
        self.board[index].update(tile)

    # Start of methods that implement QAbstractTableModel interface
    def rowCount(self, parent: QtCore.QModelIndex = ...) -> int:
        return self.board_size

    def columnCount(self, parent: QtCore.QModelIndex = ...) -> int:
        return self.board_size

    def data(self, index: QtCore.QModelIndex, role: int = ...) -> typing.Any:
        if (
            0 <= index.row() < self.rowCount()
            and 0 <= index.column() < self.columnCount()
            and index.isValid()
        ):
            if role == QMLBoard.TileRole:
                return self.board[index.column() + self.board_size * index.row()]
            else:
                raise ValueError(f"Invalid role requested: {role}")

    def roleNames(self) -> typing.Dict[int, QtCore.QByteArray]:
        roles = dict()
        roles[QMLBoard.TileRole] = b"tile"
        return roles

    # End of methods that implement QAbstractTableModel interface
    @QtCore.Slot()
    def reset(self) -> None:
        self.beginResetModel()
        self._doReset()
        self.endResetModel()

    def _doReset(self) -> None:
        """
        Reset board
        """
        self.board = [QMLTile() for _ in range(self.board_size ** 2)]

    def getTile(self, row: int, column: int) -> QMLTile:
        return self.board[column + self.board_size * column]
