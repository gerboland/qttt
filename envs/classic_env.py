import gym
from .classic_board import ClassicBoard
from copy import copy
from utils import swap_player


class ClassicTicTacToeEnv(gym.Env):
    name = "classic"
    metadata = {"render.modes": None}  # uses external renderer only

    def __init__(self, board_size: int) -> None:
        super().__init__()

        self.board = ClassicBoard(board_size)
        self.board_size = board_size

        tile_count = board_size ** 2
        self.action_space = gym.spaces.Discrete(tile_count)
        self.observation_space = gym.spaces.Discrete(tile_count)
        self.starting_player = "X"

        self.reset()

    def reset(self):
        """Resets the environment to an initial state and returns an initial observation."""
        self.board.reset()
        self.player = self.starting_player
        self.done = False
        return self.observation()

    def step(self, action: int):
        """Run one timestep of the game.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (int): position chosen by the agent

        Returns:
            observation (object): agent's observation of the game environment after the step was performed
            reward (float) : amount of reward returned after previous action
            done (bool): whether the game has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        assert self.board.valid_action(
            action
        ), f"Supplied action '{action}' was invalid"

        if self.done:
            return self.observation(), 0, True, None

        # player places her mark
        new_board = copy(self.board)  # make copy to avoid overwriting
        new_board.apply_action(self.player, action)

        reward = 0
        score = new_board.get_score(self.player)

        if score != 0:  # i.e. game is over
            self.done = True
            reward = score

        # change player
        self.player = swap_player(self.player)
        self.board = new_board
        return self.observation(), reward, self.done, None

    def available_actions(self):
        """List of actions/positions that are still free"""
        return self.board.available_actions()

    def close(self):
        """Garbage collect on close"""
        pass

    def current_player(self) -> str:
        """Return the player whose turn it is, X/O"""
        return self.player

    def set_starting_player(self, player: int) -> None:
        self.starting_player = player

    def observation(self):
        return self.board, self.player

    def board(self) -> ClassicBoard:
        return self.board

    # Human helping bits
    def help_human(self) -> str:
        return f"Tic-Tac-Toe! The goal is to get {self.board_size} in a row."

    def ask_human(self) -> str:
        return f"Enter move [1-{self.board_size ** 2}]"

    def validate_human_input(self, move: str, available_actions) -> int:
        try:
            action = int(move) - 1
        except ValueError:
            raise ValueError(
                f'Invalid input "{move}", please enter digit [1-{self.board_size**2}]'
            )

        if action not in available_actions:
            raise ValueError(f"Move {move} is already taken, please try another")

        return action
