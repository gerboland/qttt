import typing
from copy import deepcopy
from board_interface import BoardInterface
from utils import make_string_of_board
from .classic_tile import ClassicTile
import PySide2.QtCore as QtCore

EMPTY = " "


class ClassicBoard(BoardInterface):
    def __init__(
        self, board_size, state=None, parent: typing.Optional[QtCore.QObject] = None
    ) -> None:
        """
        Creates a Board for the Classic TicTacToe game.
        Arguments:
            board_size: int - if this is 3, then board is 3x3
        """
        self.board_size = board_size
        # Pre-calculate the indices of the tiles to check for winning state
        self.indices_to_check = self.generate_winning_states(board_size)
        self.winning_line = None

        super().__init__(parent)

        self.do_reset()

        if state:
            assert len(state) == self.board_size ** 2
            self.board = state

    def __copy__(self):
        board_copy = deepcopy(self.board)
        return ClassicBoard(self.board_size, board_copy)

    def hash(self) -> str:
        return "".join(map(str, self.board))

    def __str__(self) -> str:
        """
        Generates human suitable string showing the board state
        """
        return make_string_of_board(self.board, self.winning_line)

    def do_reset(self) -> None:
        self.board = [EMPTY] * (self.board_size ** 2)

    def apply_action(self, player: str, action: int) -> None:
        """
        Apply the action to the Board.
        Arguments:
            player: "X" or "O"
            action: in this case, the tile index.
        """
        index = action
        self.board[index] = player

    def available_actions(self) -> typing.List[int]:
        """
        Returns list of indices of board tiles that are open for play
        """
        return [i for i, v in enumerate(self.board) if v == EMPTY]

    def valid_action(self, action: int) -> bool:
        """
        Returns true if supplied action is permitted, else false.
        """
        return action in self.available_actions()

    def game_over(self) -> bool:
        """
        Return true if board has no moves left to play, i.e. game done. Else false
        """
        for tile in self.board:
            if tile == EMPTY:
                return False  # move is available to play

        return True

    def get_score(self, player) -> float:
        for indices in self.indices_to_check:
            line = [self.board[i] for i in indices]
            if self._matching_classical(line):
                winner = line[0]
                if winner == "X":
                    return 1.0  # for X
                elif winner == "O":
                    return -1.0  # for O

        if self.game_over():
            return 0.5  # small reward for cat's game ending

        return 0  # still in progress
