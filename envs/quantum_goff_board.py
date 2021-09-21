import typing
from copy import deepcopy
import random
import PySide2.QtCore as QtCore
import networkx as nx
from itertools import combinations

from board_interface import BoardInterface
from utils import make_string_of_board


class QuantumGoffBoard(BoardInterface):
    def __init__(
        self, board_size, state=None, parent: typing.Optional[QtCore.QObject] = None
    ) -> None:
        """
        Creates a Board for a Quantum TicTacToe game based on Goff's rules.
        Moves permitted are Superpositions, Entanglements and Classic moves.

        A cell may only contain a classical value or at most two superposition states!

        Collapse happens when there is a closed cycle of entangled tiles.
        Arguments:
            board_size: int - if this is 3, then board is 3x3

        Under the hood, am using a graph of 3x3 nodes. Classical
        states are encoded by the node having a property `player` set to
        the relevant mark `X` or `O` and a move count. Superposition states
        are are defined by an edge between the nodes, with same properties.

        A collapse considers all cycles, randomly chooses one of the 2 nodes
        and assigns the player mark to the node, making it classical.
        """
        self.board_size = board_size
        # Pre-calculate the indices of the tiles to check for winning state
        self.indices_to_check = self.generate_winning_states(board_size)

        self.G = nx.Graph()
        self.move = 0
        self.winning_line = []

        super().__init__(parent)  # calls do_reset() which creates the board

        if state:
            assert len(state) == self.board_size ** 2
            self.G = state

    def __copy__(self):
        graph_copy = deepcopy(self.G)
        new_board = QuantumGoffBoard(self.board_size, graph_copy)
        new_board.move = self.move
        return new_board

    def get_tile_value(self, index, sort=False):
        edge = self.G.adj[index]

        classic = self.get_classic(index)
        if classic:
            label = classic
        elif edge.items():
            moves = [
                f"{edge_dict['player']}{edge_dict['move']}"
                for _, edge_dict in edge.items()
            ]
            if sort:
                moves = sorted(moves)
            label = "|".join(moves)
        else:
            label = ""
        return label

    def __str__(self) -> str:
        """
        Generates human suitable string showing the board state
        """
        board = [self.get_tile_value(n) for n in self.G.nodes]
        return make_string_of_board(board, self.winning_line)

    def do_reset(self) -> None:
        self.move = 1
        self.winning_line = []
        self.G.clear()
        self.G.add_nodes_from(range(self.board_size ** 2))

    def set_value(self, index, player):
        """
        Set board tile to a classical state
        Arguments:
            index: in this case, the tile index.
            player: "X" or "O"
        """
        node = self.G.nodes[index]
        node["player"] = player
        node["move"] = self.move
        self.move += 1
        # self.tile_changed.emit(index, tile)

    def get_value(self, index):
        node = self.G[index]
        if "player" in node:
            return (node["player"], node["move"])
        else:
            None

    def collapsible_cycles_exist(self) -> bool:
        if self.game_over():  # all states are classical/collapsed already
            return False

        cycles = list(nx.cycle_basis(self.G))

        return len(cycles) > 0

    def set_superposition(self, i, j, player):
        """
        Set board tile to a classical state
        Arguments:
            i, j: the tile indices to make superposition of.
            player: "X" or "O"
        """
        assert i != j
        if i > j:  # order relied upon
            i, j = j, i

        # Check it doesn't already exist!
        try:
            assert self.G.adj[i][j], f"Superposition already exists between {i} and {j}"
        except:
            pass

        for n in [i, j]:
            assert (
                "player" not in self.G.nodes[n]
            ), f"Tile {n} is classic, you cannot place superposition there"

        self.G.add_edge(i, j, player=player, move=self.move)
        self.move += 1
        # self.tile_changed.emit(index, tile)

    def collapse_all_cycles(self):
        # Superpositions are encoded as edges in the graph.

        cycles = list(nx.cycle_basis(self.G))
        # For each cycle, collapse one edge (superpos) into its node (classic).

        for cycle in cycles:
            # Take first node, choose one of its edges to be its collapsed value
            node_id = cycle[0]
            adj_nodes = list(self.G.adj[node_id].keys())
            adj_node_id = random.choice(adj_nodes)
            edge_values = self.G.adj[node_id][adj_node_id]
            # print(f"Collapsing {node_id} to { edge_values }")
            self.G.nodes[node_id]["player"] = edge_values["player"]
            self.G.nodes[node_id]["move"] = edge_values["move"]
            self.G.remove_edge(node_id, adj_node_id)

        # Now fix up all the remaining edges & nodes
        need_iteration = True
        while need_iteration:
            need_iteration = False
            for edge in self.G.edges:
                edge_data = self.G.adj[edge[0]][edge[1]]
                node_a = self.G.nodes[edge[0]]
                node_b = self.G.nodes[edge[1]]

                if "player" in node_a and "player" in node_b:
                    # edge needs to be removed
                    self.G.remove_edge(*edge)
                    need_iteration = True

                elif "player" not in node_a and "player" in node_b:
                    node_a["player"] = edge_data["player"]
                    node_a["move"] = edge_data["move"]
                    self.G.remove_edge(*edge)
                    need_iteration = True

                elif "player" not in node_b and "player" in node_a:
                    node_b["player"] = edge_data["player"]
                    node_a["move"] = edge_data["move"]
                    self.G.remove_edge(*edge)
                    need_iteration = True

        # self.tile_changed.emit(index, tile)

    def available_actions(self) -> typing.List[int]:
        """
        Returns list of actions that are open for play
        """
        classic_tiles = []  # contain only collapsed state tiles
        quantum_tiles = []  # tiles either unset or part of superposition

        # Classical actions available
        for i in self.G.nodes:
            # not a classical state (node has data set)
            if not self.G.nodes[i] and not self.G.adj[i]:
                classic_tiles.append(i)
            if not self.G.nodes[i]:
                quantum_tiles.append(i)

        # Superpositions encoded as (i,j). First generate all possible...
        all_superposition_actions = [
            items for items in combinations(quantum_tiles, r=2)
        ]
        # ...then remove superpositions that are already set
        available_superposition_actions = list(
            filter(lambda i: i not in self.G.edges, all_superposition_actions)
        )

        return classic_tiles + available_superposition_actions

    def apply_action(self, player: str, action: int):
        if isinstance(action, int):
            self.set_value(action, player)
        else:
            self.set_superposition(*action, player)

    def valid_action(self, action: int) -> bool:
        """
        Returns true if supplied action is permitted, else false.
        """
        return action in self.available_actions()

    def game_over(self) -> bool:
        """
        Return true if board has no moves left to play, i.e. game done. Else false
        """
        return len(self.available_actions()) == 0

    def hash(self) -> int:
        """
        Hash string representing the board state. In this case, simple
        """
        tile_list = [
            self.get_tile_value(i, sort=True) if self.get_tile_value(i) else "-"
            for i in range(self.board_size ** 2)
        ]
        return "".join(tile_list)

    def _set_as_winning(self, indices: typing.List[int]) -> None:
        """
        Sets the winning property on all the supplied Tiles to true
        """
        for i in indices:
            tile = self.G.nodes[i]
            tile["winning"] = True
            self.tile_changed.emit(i, tile)

    def get_classic(self, index):
        """
        If tile has classic move, get the player that set it. Otherwise None.
        """
        return (
            self.G.nodes[index]["player"] if "player" in self.G.nodes[index] else None
        )

    def get_last_move(self, indices):
        moves = []
        for i in indices:
            if "move" in self.G.nodes[i]:
                moves.append(self.G.nodes[i]["move"])
        if len(moves) == 0:
            return 0
        return max(moves)

    def get_superposition_partner(self, index):
        """
        Returns the superposition partner of this tile, and the player that set it.
        If no superposition, returns None
        """
        if self.G.adj[index]:
            adj = self.G.adj[index]
            partner_node_index = list(adj)[0]
            player = adj.get(partner_node_index)["player"]
            return partner_node_index, player
        else:
            return None

    def get_score(self, player) -> float:
        winners = []  # in quantum game, possible to have both players win!
        for indices in self.indices_to_check:
            line = [self.get_classic(i) for i in indices]
            if self._matching_classical(line):
                self.winning_line += indices
                winner = line[0]
                if winner == "X" or winner == "O":
                    last_winning_move = self.get_last_move(indices)
                    winners.append((winner, last_winning_move))

        if len(winners) == 1:  # single winner
            winner = winners[0][0]
            if winner == "X":
                return 1.0  # for X
            elif winner == "O":
                return -1.0  # for O
        elif len(winners) > 1:
            winners.sort(key=lambda y: y[1])  # first entry has earliest winning state
            winner = winners[0][0]
            if player == winner:
                if winner == "X":
                    return 1.0  # for X
                elif winner == "O":
                    return -1.0  # for O
            else:
                if winner == "X":
                    return 0.5  # for X
                elif winner == "O":
                    return -0.5  # for O

        if self.game_over():
            if player == "X":
                return 0.5  # for X
            else:
                return -0.5  # for O
            return 0.5  # small reward for cat's game ending
        return 0  # still in progress
