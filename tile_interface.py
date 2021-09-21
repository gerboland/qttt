from copy import deepcopy
from typing import Any


class TileInterface:
    """Keeps state for each Tile on the board"""

    def __init__(self, value=None) -> None:
        self.winning: bool = False  # if tile part of winning row
        self._value = deepcopy(value)

    @property
    def filled(self) -> bool:
        # Subclass needs to decide this
        # Displays if tile is fully filled in and can no longer be edited
        raise NotImplementedError

    @property
    def value(self) -> Any:
        # This is left flexible for subclass to decide the actual type.
        # For simple Classic game, can be int or str
        return self._value

    @value.setter
    def value(self, new_value) -> None:
        self._value = deepcopy(new_value)

    def __str__(self) -> str:
        if not self.value or self.value == []:
            return " "
        return self._value
