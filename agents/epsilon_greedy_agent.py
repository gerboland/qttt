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


def keys_of_best_values(dict, min_or_max) -> typing.List[int]:
    """
    For a dictionary of the form {0: 1.0, 1: 1.0, 2: 1.0, 3: 2.0, 4: 1.0, 6: 2.0, 7: 1.0, 8: 1.0}
    it returns a list of keys with the min/max associated values, [3, 6] in this case.
    """
    best_value = min_or_max(dict.values())
    return [k for k, v in dict.items() if v == best_value]


def hash_state(state) -> str:
    """
    For 'state' which is a tuple of Board and Player, generate a string that encodes its state.
    """
    board, player = state
    return player + "-" + board.hash()


class EpsilonGreedyAgent(object):
    type = "epsilon_greedy"
    # Using shared memory between agents of this instance enables self-play training
    Q = {}

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
            self.epsilon = 0
            self.gamma = 0
        else:
            # Need to check options Dict has the options we need.
            self.alpha = read_float_arg(options, "alpha", player)
            self.epsilon = read_float_arg(options, "epsilon", player)
            self.gamma = read_float_arg(options, "gamma", player)
            # Check alpha \in [0,1], epsilon>0...

            print(
                f"Parameters for player {self.player}: alpha={self.alpha}, epsilon={self.epsilon}, gamma={self.gamma}"
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
        e = random.random()
        if e < self.epsilon * self.episode_rate:
            action = self.random_action(available_actions)
        else:
            action = self.greedy_action(state, available_actions)
        return action

    def random_action(self, available_actions) -> int:
        """ """
        return random.choice(available_actions)

    def greedy_action(self, state, available_actions) -> int:
        """
        Return best action using the Q-learning algorithm. A matrix of state/action weights
        is consulted to decide the best action to perform.
        Args:
            state (Tuple[player, board]): Game state
            available_actions (list): Available actions

        Returns:
            int: Selected action
        """
        assert len(available_actions) > 0

        state_hash = self.get_state_hash(state, available_actions)
        Qs = self.Q[state_hash]

        if self.player == "X":
            return self.get_best_action(Qs, max)
        elif self.player == "O":
            return self.get_best_action(Qs, min)

    def get_state_hash(self, state, available_actions) -> str:
        """
        Make a dictionary key for the current state (board + player turn) and if Q does not yet have it,
        add it to Q. Returns an encoding hash of the game state to be used as keys in the Q matrix.
        """
        default_Qvalue = 1.0  # Encourages exploration
        state_hash = hash_state(state)
        if state_hash not in self.Q:
            # for this board state. populate with an action weight for each possible action
            self.Q[state_hash] = {
                action: default_Qvalue for action in available_actions
            }  # The available actions in each state are initially given a default value of 1
        return state_hash

    def get_best_action(self, Qs, min_or_max) -> int:
        """
        For list of Q values for this state, find the best (i.e. min or max) action.
        Args:
            Qs: Dict[int, float] - the Q values (action weights) for a particular game state
            min_or_max           - function 'min' or 'max'
        """
        best_actions = keys_of_best_values(Qs, min_or_max)

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
        new_state_hash = self.get_state_hash(new_state, new_board.available_actions())

        if new_board.game_over():
            expected = reward
        else:
            nextQs = self.Q[new_state_hash]
            if player == "X":
                expected = reward + self.gamma * min(nextQs.values())
            else:
                expected = reward + self.gamma * max(nextQs.values())

        change = self.alpha * (expected - self.Q[state_hash][action])
        self.Q[state_hash][action] += change

    def export_model(self):
        """
        Return internal model state for archival
        """
        # some keys are defined by tuples which json will not encode, so need to transform them
        fixed_Q = {}
        for state in self.Q:
            fixed_Q[state] = {str(k): v for k, v in self.Q[state].items()}

        return dict(
            type=self.type,
            Q=fixed_Q,
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

            self.Q = data["model"]["Q"]
