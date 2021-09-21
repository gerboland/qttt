import gym
from copy import copy
from utils import swap_player

from .pure_quantum_board import PureQuantumBoard


class QuantumTicTacToe_Qubits_Env(gym.Env):
    name = "quantum_super"
    metadata = {"render.modes": None}  # uses external renderer only

    def __init__(self, board_size: int, total_moves=6) -> None:
        super().__init__()

        self.board = PureQuantumBoard(board_size)
        self.board_size = board_size
        self.total_moves = total_moves

        action_count = len(self.board.available_actions())

        self.action_space = gym.spaces.Discrete(action_count)
        self.observation_space = gym.spaces.Discrete(action_count)  # WRONG
        self.starting_player = "X"

        self.reset()

    def reset(self):
        """Resets the environment to an initial state and returns an initial observation."""
        self.board.reset()
        self.player = self.starting_player
        self.done = False
        self.move = 0
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

        reward = 0

        self.move += 1
        if self.move >= self.total_moves:
            # Collapse board to finish the game and get the score
            new_board.collapse()
            reward = new_board.get_score(self.player)
            self.done = True

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

    def board(self) -> PureQuantumBoard:
        return self.board

    # Human helping bits
    def help_human(self) -> str:
        return (
            f"Quantum Tic-Tac-Toe! The goal is to get {self.board_size} in a row.\n"
            f"Possible moves are applying X or H gates to single qubits, or CNOT to a pair\n"
            f"Set X gate of qubit 4 with 'X4', and CNOT as 'CX2,5'"
        )

    def ask_human(self) -> str:
        return f"Enter move 'Xi' or 'Hi', or 'CXi,j' where i,j are in range [1-{self.board_size**2}]"

    def validate_human_input(self, move: str, available_actions):
        try:
            move = move.lower()
            if move[0] == "x" or move[0] == "h":
                operation = move[0]
                qubit = int(move[1:])
            elif move[0:2] == "cx":
                operation = "cx"
                digits = move[2:]
                if "," not in digits:
                    raise ValueError()
                superpos = digits.split(",")
                if len(superpos) > 2:
                    raise ValueError()
                first = int(superpos[0]) - 1
                second = int(superpos[1]) - 1
                if (
                    first == second
                    or first > self.board_size ** 2
                    or second > self.board_size ** 2
                ):
                    raise ValueError()
                qubit = (first, second)
            else:
                raise ValueError()

            action = (operation, qubit)

        except ValueError:
            raise ValueError(f'Invalid input "{move}", please try again')

        if action not in available_actions:
            raise ValueError(f"Move {move} is already taken, please try another")

        return action
