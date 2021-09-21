from tile_interface import TileInterface


class SuperPositionTile(TileInterface):
    def __init__(self, node_factory) -> None:
        self.G, self.i = node_factory()
        # can contain any number of superposition states
        super().__init__()
        self._value = self.G.node[self.i]["value"]

    @property
    def pure(self) -> bool:
        return bool(self.G.node[self.i]["value"])

    def __str__(self) -> str:
        if self.pure:
            return self.G.node[self.i]["value"]
        else:
            # Get all attached nodes
            connected_nodes = self.G.neighbors[self.i]
            marks = []
            for node in connected_nodes:
                marks.append((node["player"], node["move_count"]))
            marks.sort(key=lambda tup: tup[1])  # sort by move count

            return ",".join(f"{m['player']}{m['move_count']}" for m in marks)

    @property
    def filled(self):
        return self._value == "X" or self._value == "O"
