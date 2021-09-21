from __future__ import annotations
import numpy as np
import json
import sys
import typing
from ast import literal_eval
from qiskit import QuantumCircuit, execute
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import GroverOperator


def read_float_arg(options, arg_name: str, player: str):
    try:
        return float(options[arg_name])
    except ValueError as e:
        sys.exit(
            f"Error: required argument '{arg_name}' for player '{player}' is not a float: {e}\n"
            f"Specify like this: --player{player}=ai:{arg_name}=0.4"
        )
    except TypeError as e:
        sys.exit(
            f"Error: required argument '{arg_name}' for player '{player}' is missing\n"
            f"Specify like this: --player{player}=ai:{arg_name}=0.4"
        )


def read_arg(options, arg_name: str, player: str):
    try:
        return options[arg_name]
    except ValueError as e:
        sys.exit(
            f"Error: required argument '{arg_name}' for player '{player}' is missing\n"
            f"Specify like this: --player{player}=ai:{arg_name}=file_name.model"
        )


def number_of_bits_in(input: int):
    """
    For an number n, calculate how many bits are needed to express that number.
    Note for n=1, this returns 0.
    """
    if input <= 0:
        return 0
    else:
        return int(np.ceil(np.log2(input)))


def hash_state(state) -> str:
    """
    For 'state' which is a tuple of Board and Player, generate a string that encodes its state.
    """
    board, player = state
    return player + "-" + board.hash()


class QuantumQLearningAgent(object):
    type = "quantum_qlearning"
    # Using shared memory between agents of this instance enables self-play training
    memory = {}

    def __init__(self, player, options, env):
        self.player = player
        self.options = options
        self.env = env

        self.available_actions_count = len(env.available_actions())
        # To encode the actions into qubits, need to boolean-ize them. Optimum number of
        # qubits required to encode is the log base 2 of the action count

        # Create simulator for the quantum circuits created that model the state-action states
        self.qm_simulator = AerSimulator(method="density_matrix")

        # Load an existing model?
        if "model_file" in options:
            if "alpha" in options or "epsilon" in options:
                sys.exit("Error: cannot combine model_file with epsilon or alpha")

            self.load_model(options["model_file"])

            print(
                f"AI Player '{player}' is using model loaded from '{options['model_file']}'"
            )
            self.alpha = 0
            self.k = 0
            self.gamma = 0
        else:
            # Need to check options Dict has the options we need.
            self.alpha = read_float_arg(options, "alpha", player)
            self.k = read_float_arg(options, "k", player)
            self.gamma = read_float_arg(options, "gamma", player)

            assert self.k > 0, "k must be positive"
            assert self.alpha > 0, "alpha must be positive"
            assert self.gamma > 0, "gamma must be positive"

            print(
                f"Parameters for player {self.player}: alpha={self.alpha}, k={self.k}, gamma={self.gamma}"
            )

    def act(self, state, available_actions) -> int:
        """Returns choice of action by measuring the corresponding state-action
        circuit. Called by the Env.

        Args:
            state (Tuple[player, board]): Game state
            available_actions (list): Available actions

        Returns:
            int: Selected action.
        """
        if len(available_actions) == 1:  # only one action possible, so use it.
            return available_actions[0]

        state_hash = self.get_state_hash(state, available_actions)

        action_circuit = self.memory[state_hash]["circuit"]
        print("actions in memory", self.memory[state_hash]["actions"])
        # To not destroy the quantum state, make copy
        action_circuit_copy = action_circuit.copy()
        # Apply Measurement operation to collapse the quantum states to classical probabilities
        # for the actions
        action_circuit_copy.measure_all()

        # print(action_circuit_copy.draw())

        # Perform circuit simulation. Need to loop as the measurement may return action ids that
        # are beyond the size of the available_actions (since we use base-2 encoding of action ids)
        while True:
            job = execute(action_circuit_copy, backend=self.qm_simulator, shots=1)
            result = job.result()
            counts = result.get_counts()

            # has decided state encoded as binary string like '000101010'
            state_str = list(counts.keys())[0]
            action_id = int(state_str, 2)
            print("Choose", action_id)
            print(action_circuit.draw())
            if action_id < len(available_actions):  # Good action, use it!
                break

        return available_actions[action_id]

    def get_state_hash(self, state, available_actions) -> str:
        """
        Make a dictionary key for the current state (board + player turn) and if Q does not yet have it,
        add it to Q. Returns an encoding hash of the game state to be used as keys in the Q "matrix".
        """
        state_hash = hash_state(state)

        # To encode the actions into qubits, need to boolean-ize them. Optimum number of
        # qubits required to encode is the log base 2 of the action count
        actions_qubits_count = number_of_bits_in(len(available_actions))

        if state_hash not in self.memory:
            # For this board state, populate a quantum circuit that encodes the possible actions.
            # Then place each qubit in a superposition state by applying the Hadamard gate
            # To decide an action, this circuit is measured. As it is right now, there is an
            # equal probability of any action being decided. Training amplified the amplitude (and
            # thus probability) of particular actions by applying a Grover operator to the relevant
            # qubits multiple times.
            # Note that we limit number of times this is done to avoid costly infinitesmial changes.

            if actions_qubits_count > 0:
                circuit = QuantumCircuit(actions_qubits_count)
                circuit.h(list(range(actions_qubits_count)))

                # To encode the actions into qubits, need to boolean-ize them. Optimum number of
                # qubits required to encode is the log base 2 of the action count
                binary_actions_count = number_of_bits_in(len(available_actions))

                # The maximum number of Grover steps (L) for this action for the duration of the run
                # is defined by eqn 37, 40 and following paragraph of the QRL paper by Dong,Chen,et al. 2008.
                # This appears to be designed to prevent infinitesmially small action probabilities emerging
                # as the first few Grover iterations amplify the first action too much.
                max_grover_steps = int(
                    round(
                        np.pi
                        / (4 * np.arcsin(1.0 / np.sqrt(2 ** binary_actions_count)))
                        - 0.5
                    )
                )

                # Associate a q value with the state, as well as circuit with the state, plus keep
                # track of the Grover operation count and limiter. Save the actions too.
                self.memory[state_hash] = {
                    "state_value": 1.0,  # this is not a q-value! It's a state value.
                    "circuit": circuit,
                    "grover_count": np.zeros(len(available_actions), dtype=np.int),
                    "grover_max_count": max_grover_steps,
                    "actions": available_actions,
                }

            else:  # there is only a single possible action
                self.memory[state_hash] = {
                    "state_value": 1.0,  # this is not a q-value! It's a state value.
                    "actions": available_actions,
                }

        return state_hash

    def train(self, state, new_state, reward, action) -> None:
        """
        Train the model by observing the old and new states for the reward, and use the
        Q-learning algorithm to update the Q-values for the state.
        Args:
            state       - game state before action
            new_state   - game state after action
            reward      - reward for game state after action
            action      - the action applied
        """
        board, player = state
        state_hash = self.get_state_hash(state, board.available_actions())
        new_board, new_player = new_state
        new_state_hash = self.get_state_hash(new_state, new_board.available_actions())

        this_player_reward = reward
        if reward == 1.0:
            this_player_reward = 1.0 if player == "X" else -1.0
        elif reward == -1.0:
            this_player_reward = 1.0 if player == "O" else -1.0

        # Update the state value V(s) <- V(s) + \alpha ( reward + \gamma * V(s') - V(s) )
        state_mem = self.memory[state_hash]
        state_mem["state_value"] += self.alpha * (
            this_player_reward
            + self.gamma * self.memory[new_state_hash]["state_value"]
            - state_mem["state_value"]
        )

        # Shortcut - if only one action possible, no further actions necessary
        if len(state_mem["actions"]) == 1:
            return

        # Update the number of times Grover can be applied for this action
        times_to_apply_grover = int(
            self.k * (this_player_reward + self.memory[new_state_hash]["state_value"])
        )

        # In the QRL paper by Dong,Chen,et al. 2008, they limit the number of times Grover
        # is applied in some states. This largely helps in situations with few actions,
        # where asking Grover to amplify the state too often causes them to be negligible.
        action_index = state_mem["actions"].index(action)
        grover_count_possible = (
            state_mem["grover_max_count"] - state_mem["grover_count"][action_index]
        )

        L = min(times_to_apply_grover, grover_count_possible)

        if L > 0:
            self.apply_grover_operator(state_hash, action_index, count=L)
        else:
            # print("Grover limit applied")
            pass

    def apply_grover_operator(self, state_hash, action_index, count):
        state = self.memory[state_hash]

        available_actions = state["actions"]
        binary_actions_count = number_of_bits_in(len(available_actions))

        # if not grover_max_count_hit:
        circuit = state["circuit"]

        # Convert the action index to its binary representation
        index_as_binary_str = bin(action_index)[2:].zfill(binary_actions_count)

        # Create a quantum state vector of this form for as the oracle by Grover. Essentially
        # this tells Grover that this is the "good" state and thus will amplify its amplitude
        # in the circuit.
        action_oracle_state = Statevector.from_label(index_as_binary_str)
        # Create the op
        grover_operator = GroverOperator(
            oracle=action_oracle_state, name=f"Grover [{index_as_binary_str}]"
        )

        for _ in range(count):
            # Apply the op to all qubits in the circuit. This will result in the probability
            # of the action to be increased
            circuit.append(grover_operator, list(range(binary_actions_count)))

        state["circuit"] = circuit

    def export_model(self):
        """
        Return internal model state for archival
        """
        # some keys are defined by tuples which json will not encode, so need to transform them
        fixed_memory = {}
        for state in self.memory:
            fixed_memory[state] = {str(k): v for k, v in self.memory[state].items()}

        # Quantum circuits need to be stringified
        for state in fixed_memory:
            if "circuit" in fixed_memory[state]:
                circuit = fixed_memory[state]["circuit"]
                fixed_memory[state]["circuit"] = circuit.qasm()

        return dict(
            type=self.type,
            memory=fixed_memory,
        )

    def load_model(self, filename) -> None:
        """
        Read model state from a file and import it.
        FIXME: tightly coupled with the model saving logic that lives elsewhere.
        """
        # Need to fix up the state action keys, they were integers but json converted them to strings when saving
        def jsonKeysToInt(x):
            if isinstance(x, dict):
                try:
                    return {literal_eval(k): v for k, v in x.items()}
                except ValueError:
                    return x
                except SyntaxError:
                    return x
            return x

        with open(filename, "rb") as f:
            data = json.load(f, object_hook=jsonKeysToInt)
            assert (
                self.type == data["model"]["type"]
            ), f"Model in file has type '{data.model['type']}' which is not loadable by this agent '{self.type}'"

            self.memory = data["model"]["memory"]

        # Quantum circuits need to be un-stringified
        for state in self.memory:
            if "circuit" in self.memory[state]:
                circuit = self.memory[state]["circuit"]
                self.memory[state]["circuit"] = QuantumCircuit.from_qasm_str(circuit)
