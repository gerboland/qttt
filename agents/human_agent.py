import sys


class HumanAgent:
    type = "human"

    def __init__(self, player, options, env):
        if options:
            sys.exit("Error: Simple Agent does not take any options!")
        self.player = player
        self.env = env
        print(env.help_human())

    def act(self, state, available_actions):
        while True:

            move = input(f"\n{self.env.ask_human()} or q to quit: ")
            if move.lower() == "q":
                print("Quitting!")
                sys.exit(0)

            try:
                action = self.env.validate_human_input(move, available_actions)
            except ValueError as e:
                print(e, "or q to quit")
                continue

            return action
