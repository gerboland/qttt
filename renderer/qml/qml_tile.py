import PySide2.QtCore as QtCore
from tile_interface import TileInterface


class QMLTile(QtCore.QObject):
    def __init__(self, parent=None):
        super(QMLTile, self).__init__(parent)
        self.reset()

    def reset(self):
        self._value = None  # Contains value of the tile, +/- 1 for classic game, string for quantum state
        self._filled = False  # Tile is fully set
        self._winning = False  # Tile is in winning row

    _valueChanged = QtCore.Signal()
    _filledChanged = QtCore.Signal()
    _winningChanged = QtCore.Signal()

    @QtCore.Property(str, notify=_valueChanged)
    def value(self) -> str:
        if not self._value:
            return ""
        return str(self._value)

    @QtCore.Property(bool, notify=_filledChanged)
    def filled(self) -> bool:
        return self._filled

    @QtCore.Property(bool, notify=_winningChanged)
    def winning(self) -> bool:
        return self._winning

    def update(self, tile: TileInterface):
        if self._value != tile.value:
            self._value = tile.value
            self._valueChanged.emit()

        if self._filled != tile.filled:
            self._filled = tile.filled
            self._filledChanged.emit()

        if self._winning != tile.winning:
            self._winning = tile.winning
            self._winningChanged.emit()
