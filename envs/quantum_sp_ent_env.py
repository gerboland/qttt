import typing
import gym
from copy import copy
from utils import swap_player


from .quantum_sp_board import QuantumSuperpositionBoard

Action = typing.Union[int, typing.Tuple[int, int]]


class QuantumTicTacToe_SuperPosition_Entanglement_Env(gym.Env):
    name = "sp+ent_quantum"
    metadata = {"render.modes": None}  # uses external renderer only

    def __init__(self, board_size: int) -> None:
        super().__init__()

        self.board = QuantumSuperpositionBoard(board_size)
        self.board_size = board_size

        tile_count = len(self.board.available_actions())
        self.action_space = gym.spaces.Discrete(tile_count)
        self.observation_space = gym.spaces.Discrete(tile_count)
        self.starting_player = "X"

        self.reset()

    def reset(self):
        """Resets the environment to an initial state and returns an initial observation."""
        self.board.reset()
        self.move = 0
        self.player = self.starting_player
        self.done = False
        return self.observation()

    def step(self, action: int):
        """Run one timestep of the game.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (int): position chosen by the agent

        Returns:
            observation (object): agent's observation of the current game environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the game has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        if self.done:
            return self._observation(), 0, True, None

        available_actions = self.available_actions()
        assert (
            action in available_actions
        ), f"Action {action} not available {available_actions}, {self.board}"

        # player places her mark
        new_board = copy(self.board)  # make copy to avoid overwriting
        new_board.apply_action(self.player, action)

        # Must collapse if all tiles have a classic and superposition filled in.
        # This is not a deliberate action, and makes game non-deterministic.
        if new_board.no_more_moves_need_collapse():
            # print(new_board)
            # print("\nCollapse!\n")
            new_board.collapse_all_superpositions()

        reward = 0
        # Is game over? If so, the reward is the winning player (X=1 or O=-1)
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

    def board(self) -> QuantumSuperpositionBoard:
        return self.board

    # Human helping bits
    def help_human(self) -> str:
        return f"Quantum Tic-Tac-Toe! The goal is to get {self.board_size} in a row."

    def ask_human(self) -> str:
        return f"Enter move 'x' or a superposition with 'x,y' where x,y are in range [1-{self.board_size**2}]"

    def validate_human_input(self, move: str, available_actions):
        try:
            if "," in move:
                superpos = move.split(",")
                if len(superpos) > 2:
                    raise ValueError()
                first = int(superpos[0]) - 1
                second = int(superpos[1]) - 1
                if first == second:
                    raise ValueError()
                elif first < second:
                    action = (first, second)
                else:
                    action = (second, first)

            else:
                action = int(move) - 1

        except ValueError:
            raise ValueError(f'Invalid input "{move}", please try again')

        if action not in available_actions:
            raise ValueError(f"Move {move} is already taken, please try another")

        return action
