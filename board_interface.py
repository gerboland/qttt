import typing
import PySide2.QtCore as QtCore
from functools import lru_cache
from math import sqrt
from tile_interface import TileInterface


def all_equal_values(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator).value
    except StopIteration:
        return True
    return all(first == x.value for x in iterator)


class BoardInterface(QtCore.QObject):
    board_reset = QtCore.Signal()
    tile_changed = QtCore.Signal(int, TileInterface)

    def __init__(self, parent: typing.Optional[QtCore.QObject] = None) -> None:
        """
        Shared interface for a Board, which is a collection of Tiles.
        """
        super(BoardInterface, self).__init__(parent)

        self.do_reset()

    def reset(self) -> None:
        self.do_reset()
        self.board_reset.emit()

    def do_reset(self) -> None:
        """
        Implement this to set up a new Board
        """
        raise NotImplementedError("Subclass BoardInterface and implement this!")

    def __hash__(self) -> str:
        """
        Returns a string that is a hash of the board state
        """
        raise NotImplementedError("Subclass BoardInterface and implement this!")

    def __str__(self) -> str:
        """
        Returns a string that displays the board state to be understood by a human
        """
        raise NotImplementedError("Subclass BoardInterface and implement this!")

    def apply_action(self, player, action):
        """
        Perform an action on the Board.
        Arguments:
            player: 'x' or 'o'
            action: the action to perform (implementation specific)

        Note: implementation must call self.tile_changed.emit() when a tile is changed
        so the UI will pick up the change.
        """
        raise NotImplementedError("Subclass BoardInterface and implement this!")

    def get_score(self, player) -> float:
        """
        Get score for each player
            X win = 1
            O win = -1
            Draw  = 0.5
            Game in progress: 0
        """
        raise NotImplementedError("Subclass BoardInterface and implement this!")

    def available_actions(self) -> typing.List[typing.Any]:
        """
        Returns a list of permitted actions.
        """
        raise NotImplementedError("Subclass BoardInterface and implement this!")

    def valid_action(self, action) -> bool:
        """
        Returns true if supplied action is permitted, else false.
        """
        raise NotImplementedError("Subclass BoardInterface and implement this!")

    def game_over(self) -> bool:
        """
        Are any moves remaining?
        """
        raise NotImplementedError("Subclass BoardInterface and implement this!")

    @staticmethod
    @lru_cache
    def generate_winning_states(board_size: int):
        """
        Get the list of indices of the tiles to check for winning state
        """
        indices = []
        for i in range(board_size):
            # generate indices of the rows
            indices.append([j + board_size * i for j in range(board_size)])
            # columns
            indices.append([i + board_size * j for j in range(board_size)])

        # diagonals
        indices.append([i + board_size * i for i in range(board_size)])
        indices.append(
            [(board_size - i - 1) + board_size * i for i in range(board_size)]
        )
        return indices

    @staticmethod
    def _matching_classical(lst: typing.List[int]) -> bool:
        """
        Returns True if list has all equal integer values
        """
        for l in lst:
            if l != "X" and l != "O":
                return False  # not X,O

        # ok, all have classical value, check they match
        def all_equal(iterator):
            iterator = iter(iterator)
            try:
                first = next(iterator)
            except StopIteration:
                return True
            return all(first == x for x in iterator)

        return all_equal(lst)

    # @staticmethod
    # def _matching_classical(tiles: typing.List[int]) -> bool:
    #     """
    #     Returns True if list has all equal classical states (i.e. X,O)
    #     """
    #     for t in tiles:
    #         if t != "X" and t != "O":
    #             return False  # not X,O

    #     # ok, so all filled in, and have classical value, check they match
    #     return all_equal_values(tiles)
