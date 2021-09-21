class NullRenderer:
    def render(board) -> None:
        # board change
        # emit boardChanged()
        pass

    def prompt_player(player: int) -> None:
        # signal game to ask for user input?
        # emit playerChanged(player)
        pass

    def game_over(player, reward) -> None:
        # emit gameOver(player)
        pass
