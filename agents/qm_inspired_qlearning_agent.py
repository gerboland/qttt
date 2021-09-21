from __future__ import annotations
import random
import json
import sys
import typing
from ast import literal_eval


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


def best_q_value(state_memory, min_or_max) -> float:
    """
    Returns the best (min/max) q-value in the state_memory.
    Args:
        state_memory - a Dict of the form {action: (q-value, probability)}
    """
    return min_or_max(state_memory.values(), key=lambda x: x[0])[0]


def normalise_probabilities(state_memory) -> None:
    """
    Normalises the probabilities for all the possible actions in the state memory.
    Performs the normalisation to the input data structure.
    Args:
        state_memory - a Dict of the form {action: (q-value, probability)}
    """
    total = sum(list(zip(*state_memory.values()))[1])

    for (key, q_p_tuple) in state_memory.items():
        state_memory[key] = (q_p_tuple[0], q_p_tuple[1] / total)


def hash_state(state) -> str:
    """
    For 'state' which is a tuple of Board and Player, generate a string that encodes its state.
    """
    board, player = state
    return player + "-" + board.hash()


class QuantumInspiredQLearningAgent(object):
    type = "quantum_inspired_qlearning"
    # Using shared memory between agents of this instance enables self-play training
    memory = {}

    def __init__(self, player, options, env):
        self.player = player
        self.options = options
        self.env = env

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
            # Check alpha \in [0,1], k>0...

            print(
                f"Parameters for player {self.player}: alpha={self.alpha}, k={self.k}, gamma={self.gamma}"
            )

        self.episode_rate = 1.0  # Set during training, decreases...

    def act(self, state, available_actions) -> int:
        """Returns action by Epsilon greedy policy. Called by the Env.

        Return random action with epsilon probability or best action.

        Args:
            state (Tuple[player, board]): Game state
            available_actions (list): Available actions

        Returns:
            int: Selected action.
        """
        action = self.prob_action(state, available_actions)
        return action

    def prob_action(self, state, available_actions) -> int:
        """
        Return best action using the Quantum inspired Q-learning algorithm. A matrix of state/action
        probabilities is consulted to decide the best action to perform.
        Args:
            state (Tuple[player, board]): Game state
            available_actions (list): Available actions

        Returns:
            int: Selected action
        """
        assert len(available_actions) > 0

        state_hash = self.get_state_hash(state, available_actions)
        state_memory = self.memory[state_hash]

        available_action_probabilities = [
            state_memory[action][1] for action in available_actions
        ]

        # Sample from the distribution of actions
        action_choice = random.choices(
            available_actions, weights=available_action_probabilities, k=1
        )[0]
        return action_choice

    def get_state_hash(self, state, available_actions) -> str:
        """
        Make a dictionary key for the current state (board + player turn) and if Q does not yet have it,
        add it to Q. Prefill it with default values for Q and the probability P.

        Returns an encoding hash of the game state to be used as keys in the memory to retrieve Q and P.
        """
        default_Qvalue = 0
        default_prob = (
            (1.0 / len(available_actions)) if len(available_actions) > 0 else 1.0
        )

        # as we require sum of all actions to be 1
        state_hash = hash_state(state)
        if state_hash not in self.memory:
            # for this board state, populate with an action weight and probability for each possible action
            action_tuple = (default_Qvalue, default_prob)

            self.memory[state_hash] = {
                action: action_tuple for action in available_actions
            }
        return state_hash

    def get_best_action(self, state_memory, min_or_max) -> int:
        """
        For list of Q values for this state, find the best (i.e. min or max) action.
        Args:
            state_memory: Dict[int, Tuple[q: float, p: float]] - the P & Q values (action weights) for a particular game state
            min_or_max           - function 'min' or 'max'
        """
        best_value = min_or_max(state_memory.values(), key=lambda x: x[0])[0]
        best_actions = [k for k, v in dict.items() if v[0] == best_value]

        if len(best_actions) > 1:
            # There is more than one action corresponding to the min/maximum Q-value, choose one at random
            return random.choices(best_actions, k=1)[0]
        else:
            return best_actions[0]

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

        state_action_data = self.memory[state_hash]

        # Clunky
        if reward == 1.0:
            prob_reward = 1.0 if player == "X" else -1.0
        elif reward == -1.0:
            prob_reward = 1.0 if player == "O" else -1.0
        else:
            prob_reward = reward

        # Update the Q values for the state
        if new_board.game_over():
            q_expected = reward
            p_expected = prob_reward
        else:
            new_state_hash = self.get_state_hash(
                new_state, new_board.available_actions()
            )
            new_state_action_data = self.memory[new_state_hash]

            if player == "X":
                best_new_q_value = best_q_value(new_state_action_data, min)

            else:
                best_new_q_value = best_q_value(new_state_action_data, max)

            q_expected = reward + self.gamma * best_new_q_value
            p_expected = prob_reward + best_new_q_value

        change = self.alpha * (q_expected - state_action_data[action][0])
        new_q_value = state_action_data[action][0] + change

        # Update the probabilities for the state actions
        new_prob = state_action_data[action][1] + self.k * p_expected
        # Avoid negative probabilities
        if new_prob < 0:
            new_prob = 0.001

        state_action_data[action] = (new_q_value, new_prob)

        normalise_probabilities(state_action_data)

    def export_model(self):
        """
        Return internal model state for archival
        """
        # some keys are defined by tuples which json will not encode, so need to transform them
        fixed_memory = {}
        for state in self.memory:
            fixed_memory[state] = {str(k): v for k, v in self.memory[state].items()}

        return dict(
            type=self.type,
            memory=fixed_memory,
        )

    def load_model(self, filename) -> None:
        """
        Read model state from a file and import it.
        FIXME: tightly coupled with the model saving logic that lives elsewhere.
        """
        # Need to fix up the state action keys, they were integers or tuples but json converted them to strings when saving
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
