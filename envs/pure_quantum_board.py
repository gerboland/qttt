import typing
from copy import deepcopy

from board_interface import BoardInterface

import PySide2.QtCore as QtCore
import numpy as np
from qiskit import QuantumCircuit, execute
from qiskit.providers.aer import AerSimulator, StatevectorSimulator
from itertools import permutations
from utils import make_string_of_board


class PureQuantumBoard(BoardInterface):
    def __init__(
        self, board_size, state=None, parent: typing.Optional[QtCore.QObject] = None
    ) -> None:
        """
        Creates a Board for a Quantum TicTacToe game, where each tile consists of
        a qubit, which can be manipulated solely with X, H and CNOT gates.

        Collapse is performed by a player, after which the game should be over.

        Arguments:
            board_size: int - if this is 3, then board is 3x3

        """
        self.board_size = board_size
        tile_count = board_size ** 2

        # Create simulator of the quantum circuit created as Quantum TicTacToe moves are applied
        # Want to simulate a noisy circuit that we measure at the end of the game.
        self.qm_simulator = AerSimulator(method="density_matrix")

        # This simulation applied to a noisy circuit (which the game is) performs a sample of the
        # randomly sampled noisy circuit from the noise model, each time it is invoked.
        self.statevector_simulator = StatevectorSimulator(precision="single")

        # In this game, all moves remain available, so available_actions() will return a fixed array.
        # Generate it now
        self._available_actions = self._generate_available_actions(tile_count)

        # Pre-calculate the indices of the tiles to check for winning state
        self.indices_to_check = self.generate_winning_states(board_size)

        super().__init__(parent)  # calls do_reset() which creates the board

        if state:
            self.game_circuit = state

    def do_reset(self) -> None:
        self.winning_line = []
        self.collapsed_board_state = None
        tile_count = self.board_size ** 2

        # Game is modeled as a quantum circuit with `tile_count` qubit registers. Game moves result
        # in adding operations (X, H, CNOT) that apply to one or two of those registers.
        # This creates a circuit with tile_count qubits.
        self.game_circuit = QuantumCircuit(tile_count)

        # Initialise the `tile_count` qubits into a superposition state by applying the Hadamard date.
        self.game_circuit.h(list(range(tile_count)))
        return self.observation()

    @staticmethod
    def _generate_available_actions(qubit_count: int):
        """
        Generate a list of all possible actions. These include placing H or X on a single chosen
        qubit, and applying a CNOT to a chosen pair of qubits.
        Each action encoded as a Tuple: qubit(s),Gate
        """
        actions = []
        for i in range(qubit_count):
            actions.append(("h", i))
            actions.append(("x", i))
        for pair in permutations(list(range(qubit_count)), r=2):
            actions.append(("cx", pair))
        return actions

    def __copy__(self):
        game_circuit_copy = deepcopy(self.game_circuit)
        new_board = PureQuantumBoard(self.board_size, game_circuit_copy)
        return new_board

    def __str__(self) -> str:
        """
        Generates human suitable string showing the board state
        """
        qcircuit = str(self.game_circuit.draw(output="text"))

        if not self.game_over():
            return qcircuit
        else:
            return (
                "\nFinal move, board collapses!\n"
                + qcircuit
                + "\n"
                + make_string_of_board(self.collapsed_board_state)
            )

    def observation(self):
        """
        Quantumly observe the board, return it as the statevector of the board circuit
        :return: rounded state vec of the board.
        """
        job = execute(self.game_circuit, self.statevector_simulator, shot=1)
        result = job.result()
        output_state = result.get_statevector()
        return output_state

    def available_actions(self) -> typing.List[int]:
        """
        Returns list of actions that are open for play.
        """
        return self._available_actions

    def apply_action(self, player: str, action):
        """
        Apply action by appending the associated gate to the board circ.
        :param action: int, key of the moves dict
        :return:
        """
        gate_name = action[0]
        argument = action[1]
        if gate_name == "h":
            self.game_circuit.h(argument)
        elif gate_name == "x":
            self.game_circuit.x(argument)
        elif gate_name == "cx":
            self.game_circuit.cx(*argument)
        else:
            raise ValueError(f"Unknown action {action}")

    def valid_action(self, action: int) -> bool:
        """
        Returns true if supplied action is permitted, else false.
        """
        return action in self._available_actions()

    def collapse(self):
        """
        Final move, measure the board and observe the final board state
        :return: final classical state of the board as list of X & Os
        """
        self.game_circuit.measure_all()
        job = execute(
            self.game_circuit, backend=self.qm_simulator, shots=1
        )  # as noisy backend, 1 shot enough
        res = job.result()
        counts = res.get_counts()

        # has final state encoded as string like '000101010'
        state_str = list(counts.keys())[0]

        # want to return array of form [1,2,1....] where 1:=X, 2:=0
        self.collapsed_board_state = [int(i) + 1 for i in list(state_str)]
        return self.collapsed_board_state

    def hash(self) -> str:
        """
        Hash string representing the board state. Just taking a list of the operands
        is insufficient as the order the operators occur would matter. Best way is to
        perform an observation of the current board state vector.
        """
        observation_vector = self.observation()
        # try to avoid floating point noise by rounding
        return str(np.around(observation_vector, decimals=2))

    def game_over(self) -> bool:
        return bool(self.collapsed_board_state)

    def get_score(self, player) -> float:
        if not self.game_over():
            return 0

        board = self.collapsed_board_state
        winners = set()  # in quantum game, possible to have both players win!
        for indices in self.indices_to_check:
            line = [board[i] for i in indices]
            if self._matching_classical(line):
                self.winning_line = indices
                winner = line[0]
                if winner > 0:
                    winners.add(winner)

        if len(winners) == 1:  # single winner
            winner = winners.pop()
            if winner == 1:
                return 1.0  # for X
            elif winner == 2:
                return -1.0  # for O
        elif len(winners) > 1:
            return 0.7  # smaller reward for a draw where both players get a line
        else:
            return 0.5  # small reward for cat's game ending (no winners)

    @staticmethod
    def _matching_classical(lst: typing.List[int]) -> bool:
        """
        Returns True if list has all equal integer values
        """
        for l in lst:
            if l == 0:
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
