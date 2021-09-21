import typing
from copy import deepcopy
import random
import termtables as tt
from colorama import Fore, Back, Style
from .quantum_sp_board import QuantumSuperpositionBoard

import PySide2.QtCore as QtCore
import networkx as nx
from itertools import combinations


class QuantumSuperpositionEntanglementBoard(QuantumSuperpositionBoard):
    def __init__(
        self, board_size, state=None, parent: typing.Optional[QtCore.QObject] = None
    ) -> None:
        """
        Creates a Board for a simplified Quantum TicTacToe game.
        Moves permitted are Classic moves and Superpositions, which may form
        entanglements.

        A cell may only contain a classical value or at most TWO superposition state!

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
        super().__init__(
            board_size, state, parent
        )  # calls do_reset() which creates the board

    def __copy__(self):
        graph_copy = deepcopy(self.G)
        new_board = QuantumSuperpositionEntanglementBoard(self.board_size, graph_copy)
        new_board.move = self.move
        return new_board

    def no_more_moves_need_collapse(self) -> bool:
        if self.game_over():  # all states are classical/collapsed already
            return False

        # Check every tile is either classic or a quantum entanglement
        for i in self.G.nodes:
            # not a classical state (node has data set), and not more than two superpositions (edge)
            node_set = self.get_classic(i)
            edge_all_set = len(self.G.adj[i]) > 1
            if not node_set and not edge_all_set:
                return False

        return True

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
