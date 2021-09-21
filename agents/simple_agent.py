import random
import sys

from copy import copy
from utils import swap_player


def player_score(player):
    if player == "X":
        return 1.0
    else:
        return -1.0


class SimpleAgent(object):
    type = "simple_ai"

    def __init__(self, player, options, env):
        if options:
            sys.exit("Error: Simple Agent does not take any options!")
        self.player = player

    def act(self, state, available_actions):
        best_action = None
        best_score = -1
        board, _ = state

        for action in available_actions:
            # copy board state, play the move, see if it might win
            new_board = copy(board)
            new_board.apply_action(self.player, action)

            score = new_board.get_score(self.player)

            if score == player_score(swap_player(self.player)):
                # Block opponent from winning
                return action

            if score == player_score(self.player):
                # Winning move
                return action

        if best_score > 0:
            return best_action

        return random.choice(available_actions)
