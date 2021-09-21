from tile_interface import TileInterface


class ClassicTile(TileInterface):
    def __init__(self) -> None:
        super().__init__(None)  # value is None, but can be "X" or "O" strings

    @property
    def filled(self):
        return self._value == "X" or self._value == "O"
