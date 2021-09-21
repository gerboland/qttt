import typing
from copy import deepcopy
import random
import PySide2.QtCore as QtCore
import networkx as nx
from itertools import combinations

from board_interface import BoardInterface
from utils import make_string_of_board


class QuantumSuperpositionBoard(BoardInterface):
    def __init__(
        self, board_size, state=None, parent: typing.Optional[QtCore.QObject] = None
    ) -> None:
        """
        Creates a Board for a Quantum TicTacToe game.
        Moves permitted are Superposition and Classic moves.

        A cell may only contain a classical value or at most one superposition state!

        Collapse happens when all tiles have a move in them.
        Arguments:
            board_size: int - if this is 3, then board is 3x3

        Under the hood, am using a graph of 3x3 nodes. Classical
        states are encoded by the node having a property `player` set to
        the relevant mark `X` or `O` and a move count. Superposition states
        are are defined by an edge between the nodes, with same properties.

        A collapse considers all edges, randomly chooses one of the 2 nodes
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
        new_board = QuantumSuperpositionBoard(self.board_size, graph_copy)
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
        self.move = 0
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

    def no_more_moves_need_collapse(self) -> bool:
        if self.game_over():  # all states are classical/collapsed already
            return False

        # Check every tile is either classic or a quantum entanglement
        for i in self.G.nodes:
            # not a classical state (node has data set), and not in a superposition (edge)
            node_set = self.get_classic(i)
            edge_set = len(self.G.adj[i]) > 0
            if not node_set and not edge_set:
                return False

        return True

    def set_superposition(self, i, j, player):
        """
        Set board tile to a classical state
        Arguments:
            i, j: the tile indices to make superposition of.
            player: "X" or "O"
        """
        assert i != j
        self.G.add_edge(i, j, player=player, move=self.move)
        self.move += 1
        # self.tile_changed.emit(index, tile)

    def collapse_all_superpositions(self):
        # Superpositions are encoded as edges in the graph.
        for edge in self.G.edges:
            pick = random.choice([0, 1])
            # Collapse chooses one of the nodes attached to the edge
            live_cat = edge[pick]
            live_node = self.G.nodes[live_cat]

            # Copy the data in the edge to the chosen node
            edge_data = self.G.adj[edge[0]][edge[1]]
            for key, value in edge_data.items():
                live_node[key] = value

            # Delete the edge
            self.G.remove_edge(*edge)
            # self.tile_changed.emit(index, tile)

    def available_actions(self) -> typing.List[int]:
        """
        Returns list of actions that are open for play
        """
        actions = []

        # Classical actions available
        for i in self.G.nodes:
            # not a classical state (node has data set), and not in a superposition (edge)
            if not self.G.nodes[i] and not self.G.adj[i]:
                actions.append(i)

        # Superpositions encoded as (i,j)
        superposition_actions = [items for items in combinations(actions, r=2)]

        return actions + superposition_actions

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
        for i in self.G.nodes:
            # not a classical state
            if not self.get_classic(i):
                return False  # move is available to play

        return True

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
        winners = set()  # in quantum game, possible to have both players win!
        for indices in self.indices_to_check:
            line = [self.get_classic(i) for i in indices]
            if self._matching_classical(line):
                self.winning_line = indices
                winner = line[0]
                if winner == "X" or winner == "O":
                    winners.add(winner)

        if len(winners) == 1:  # single winner
            winner = winners.pop()
            if winner == "X":
                return 1.0  # for X
            elif winner == "O":
                return -1.0  # for O
        elif len(winners) > 1:
            return 0.5  # smaller reward for a draw

        if self.game_over():
            return 0.5  # small reward for cat's game ending
        return 0  # still in progress
