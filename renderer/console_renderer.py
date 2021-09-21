from colorama import Fore, Style, init
import sys

from board_interface import BoardInterface

init()  # colorama needs to do this on Windows


class ConsoleRenderer:
    def render(board: BoardInterface) -> None:
        # Board has implemented __str__ which outputs a nicely formatted string
        print(board)

    def prompt_player(player: int) -> None:
        print(f"Turn of player: {player}")

    def game_over(player, reward) -> None:
        if reward == 0:
            print(Fore.YELLOW + "GAME OVER: DRAW\n\n" + Style.RESET_ALL)
        else:
            print(
                Fore.YELLOW + f"GAME OVER: Player {player} won!\n\n" + Style.RESET_ALL
            )

        move = input("\nPress q to quit, or any key to play again... ")
        if move.lower() == "q":
            print("Quitting!")
            sys.exit(0)
