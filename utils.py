import termtables as tt
from colorama import Fore, Style
from math import sqrt


def swap_player(player: str):
    return "X" if player == "O" else "O"


def make_string_of_board(board, winning_line=[]) -> str:
    """
    Generates human suitable string showing the board state
    """
    board_size = int(sqrt(len(board)))
    data = [[None for _ in range(board_size)] for _ in range(board_size)]
    for i in range(board_size):
        for j in range(board_size):
            index = j + board_size * i
            player = board[j + board_size * i]
            if winning_line and index in winning_line:
                data[i][j] = Fore.GREEN + player + Style.RESET_ALL
            else:
                data[i][j] = player

    return tt.to_string(data)
