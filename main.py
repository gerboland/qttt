#!/usr/bin/env python
import click
import sys
import os
from tqdm import tqdm
import typing
from datetime import datetime
import json

from PySide2 import QtCore, QtGui, QtQml
from PySide2.QtCore import QObject, QThread

from envs.classic_env import ClassicTicTacToeEnv
from envs.quantum_sp_env import QuantumTicTacToe_Superposition_Env
from envs.pure_quantum_env import QuantumTicTacToe_Qubits_Env
from envs.quantum_goff_env import QuantumTicTacToe_Goff_Env
from agents.simple_agent import SimpleAgent
from agents.human_agent import HumanAgent
from agents.qml_agent import QMLAgent
from agents.epsilon_greedy_agent import EpsilonGreedyAgent
from agents.qm_inspired_qlearning_agent import QuantumInspiredQLearningAgent
from agents.quantum_qlearning_agent import QuantumQLearningAgent

# from envs.goff_quantum_env import GoffQuantumTicTacToeEnv
from renderer.console_renderer import ConsoleRenderer
from renderer.null_renderer import NullRenderer
from renderer.qml.game_context import QMLGameContext
from renderer.qml_renderer import QMLRenderer
from utils import swap_player

ai_agent_list = ["simple", "epsilon_greedy", "qiql", "quantumqlearning"]
agent_list = ["human"] + ai_agent_list
games = ["classic", "quantum_super", "quantum_goff", "qubit"]


def get_agent(player: str, id: int, options, need_qt: bool, env):
    if player == "human":
        return QMLAgent(id) if need_qt else HumanAgent(id, options, env)
    elif player == "epsilon_greedy":
        return EpsilonGreedyAgent(id, options, env)
    elif player == "qiql":
        return QuantumInspiredQLearningAgent(id, options, env)
    elif player == "quantumqlearning":
        return QuantumQLearningAgent(id, options, env)
    else:
        return SimpleAgent(id, options, env)


def get_env_for_game(game: str, board_size: int):
    # decide game
    if game == "classic":
        return ClassicTicTacToeEnv(board_size)
    elif game == "quantum_super":
        return QuantumTicTacToe_Superposition_Env(board_size)
    elif game == "quantum_goff":
        return QuantumTicTacToe_Goff_Env(board_size)
    elif game == "qubit":
        return QuantumTicTacToe_Qubits_Env(board_size)
    else:
        raise RuntimeError("Not done yet")


def key_value_string_to_dict(input: str) -> typing.Dict[str, typing.Any]:
    return dict(u.split("=") for u in input.split(","))


def validate_player_spec(ctx, param, value):
    if not isinstance(value, str):
        raise click.BadParameter(
            "player must be one of" + ",".join(map("'{0}'".format, agent_list))
        )

    try:
        name_and_opts = value.split(":", maxsplit=1)  # split on first ':'

        assert len(name_and_opts) < 3
        assert name_and_opts[0].lower() in agent_list
        if len(name_and_opts) == 1:
            name_and_opts.append(None)
        else:
            name_and_opts[1] = key_value_string_to_dict(name_and_opts[1])
        return name_and_opts

    except AssertionError:
        raise click.BadParameter(
            "format must be ["
            + "|".join(map("'{0}'".format, agent_list))
            + "]:key=value,key2=val2'"
        )


@click.group()
def cli():
    pass


@click.command(help="Play TicTacToe, classic or quantum!")
@click.option(
    "--game",
    default="classic",
    type=click.Choice(games, case_sensitive=False),
)
@click.option("--board_size", default=3, type=click.IntRange(2, 6))
@click.option(
    "-pX",
    "--playerX",
    default="human",
    type=click.UNPROCESSED,
    callback=validate_player_spec,
    help="Player specified with the form: [human|simple|ai:model_file=trained_model.dat]",
)
@click.option(
    "-pO",
    "--playerO",
    default="simple",
    type=click.UNPROCESSED,
    callback=validate_player_spec,
    help="Player specified with the form: [human|simple|epsilon_greedy|qiql:key=value,other=option]",
)
@click.option(
    "--ui",
    default="console",
    type=click.Choice(["console", "null", "qt"], case_sensitive=False),
)
def play(
    game: str,
    board_size: int,
    playerx: str,
    playero: str,
    ui: str,
):
    playerx, playerx_options = playerx
    playero, playero_options = playero
    need_qt = False

    # decide game
    env = get_env_for_game(game, board_size)

    # set renderer on the env: null, asci, UI (human only)
    if ui == "qt":
        need_qt = True
        renderer = QMLRenderer()
    elif ui == "console":
        renderer = ConsoleRenderer
    else:
        if playerx == "human" or playero == "human":  # puny humans need a UI
            print("Cannot permit human players with Null UI")
            sys.exit(1)
        renderer = NullRenderer

    # decide players
    agents = [
        get_agent(playerx, "X", playerx_options, need_qt, env),
        get_agent(playero, "O", playero_options, need_qt, env),
    ]

    if need_qt:
        # Bring up Qt, and run the game logic in a separate thread
        app = QtGui.QGuiApplication(sys.argv)

        # Create worker thread & put object on that thread
        thread = QtCore.QThread()
        thread.start()
        worker = Worker(env, agents, renderer)
        worker.moveToThread(thread)
        worker.start.connect(worker.do_play)
        worker.start.emit()  # runs 'do_play' on worker thread
        worker.finished.connect(app.quit)

        engine = QtQml.QQmlApplicationEngine()

        gameContext = QMLGameContext(env, agents, renderer)

        engine.rootContext().setContextProperty("context", gameContext)
        directory = os.path.dirname(os.path.abspath(__file__))
        engine.load(
            QtCore.QUrl.fromLocalFile(
                os.path.join(directory, "renderer", "qml", "main.qml")
            )
        )
        if not engine.rootObjects():
            return -1

        ret = (
            app.exec_()
        )  # runs Qt event loop, blocks until UI window closed or app.quit() called
        thread.quit()
        thread.wait(1)
        thread.terminate()
        return ret
    else:
        do_play(env, agents, renderer=renderer)
        return 0


class Worker(QObject):
    finished = QtCore.Signal()
    start = QtCore.Signal()

    def __init__(self, env, agents, renderer, parent=None):
        super(Worker, self).__init__(parent)
        self.env = env
        self.agents = agents
        self.renderer = renderer

    @QtCore.Slot()
    def do_play(self):
        try:
            play(
                self.env,
                self.agents,
                self.renderer,
            )
        except SystemExit:  # don't just exit as need to stop Qt event loop
            pass
        self.finished.emit()


def do_play(env, agents, renderer=NullRenderer, episodes=10):
    starting_player = "X"
    scores = {"X": 0, "O": 0}

    for _ in range(episodes):
        env.set_starting_player(starting_player)
        state = env.reset()
        while not env.done:
            player = env.current_player()
            renderer.prompt_player(player)
            agent = agents[int(player != "X")]  # get Agent
            available_actions = env.available_actions()
            action = agent.act(state, available_actions)
            state, reward, done, info = env.step(action)
            renderer.render(env.board)

        if reward == 1:
            scores["O"] += 1
        elif reward == -1:
            scores["X"] += 1

        renderer.game_over(player, reward)

        # rotate starting player
        starting_player = swap_player(starting_player)

    scores["D"] = episodes - scores["X"] - scores["O"]
    return scores


@click.command(help="Train AI Agents to learn TicTacToe, classic or quantum!")
@click.option(
    "--game",
    default="classic",
    type=click.Choice(games, case_sensitive=False),
)
@click.option("--board_size", default=3, type=click.IntRange(2, 6))
@click.option(
    "-p",
    "--player",
    default="ai",
    type=click.UNPROCESSED,
    callback=validate_player_spec,
    help="Player specified with the form: [human|simple|ai:key=value,other=option]",
)
@click.option(
    "--save_model_file", type=click.File("w"), help="Save trained model to disk"
)
@click.option("--episodes", default=1000, type=click.IntRange(1, 10000000))
def train(game, board_size, player, episodes, save_model_file):
    player, player_options = player

    env = get_env_for_game(game, board_size)

    # decide players
    agents = [
        get_agent(player, "X", player_options, False, env),
        get_agent(player, "O", player_options, False, env),
    ]

    # Ensure at least one agent is trainable
    if not hasattr(agents[0], "train"):
        sys.exit("Error: Agent is not trainable!")

    do_train(env, agents, episodes)

    # write model & metadata to file
    data = {
        "game": game,
        "board_size": board_size,
        "date": datetime.utcnow().isoformat(),
        "options": player_options,
        "model": agents[0].export_model(),
    }
    json.dump(data, save_model_file, indent=2)
    print(f"Training complete, model saved to '{save_model_file.name}'")


def do_train(env, agents, episodes=10000):
    starting_player = "X"

    for i in tqdm(range(episodes)):
        env.set_starting_player(starting_player)

        # Update Episode rate in the Agents
        for agent in agents:
            agent.episode_rate = 1 - (i + 1) / float(episodes)

        state = env.reset()
        while not env.done:
            player = env.current_player()
            agent = agents[int(player == "X")]  # get Agent
            available_actions = env.available_actions()
            action = agent.act(state, available_actions)
            newstate, reward, done, info = env.step(action)

            agent.train(state, newstate, reward, action)

            state = newstate

        # rotate starting player
        starting_player = swap_player(starting_player)


@click.command(help="Benchmark trained AI agents")
@click.option(
    "-pX",
    "--playerX",
    default="ai",
    type=click.Choice(ai_agent_list, case_sensitive=False),
)
@click.option("--playerX_model_file", type=str, help="Trained model file for Player X")
@click.option(
    "-pO",
    "--playerO",
    default="simple",
    type=click.Choice(ai_agent_list, case_sensitive=False),
)
@click.option(
    "--playerO_model_file",
    type=str,
    help="Trained model file for Player O",
)
@click.option("--episodes", default=1000, type=click.IntRange(1, 10000000))
def benchmark(episodes, playerx, playerx_model_file, playero, playero_model_file=None):
    # Start with some defaults
    game = None
    board_size = -1
    playerx_options, playero_options = None, None

    # If players not 'simple', need to check they were trained on same game and board size
    # and then create suitable game env
    if playerx != "simple":
        with open(playerx_model_file, "rb") as f:
            model = json.load(f)
            game = model["game"]
            board_size = model["board_size"]
        playerx_options = {"model_file": playerx_model_file}

    if playero != "simple":
        with open(playero_model_file, "rb") as f:
            model = json.load(f)
            if game and game != model["game"]:
                sys.exit("The supplied models were trained on different games")
            else:
                game = model["game"]

            if board_size >= 0 and board_size != model["board_size"]:
                sys.exit("The supplied models were trained on different board sizes")
            else:
                board_size = model["board_size"]
        playero_options = {"model_file": playero_model_file}

    env = get_env_for_game(game, board_size)

    print(f"Benchmarking '{game}' game with board size {board_size}*{board_size}")

    # decide players
    agents = [
        get_agent(playerx, "X", playerx_options, False, env),
        get_agent(playero, "O", playero_options, False, env),
    ]

    scores = do_benchmark(env, agents, episodes)
    print("Benchmarking results:", scores)


def do_benchmark(env, agents, episodes=3000):
    starting_player = "X"
    results = []

    for _ in tqdm(range(episodes)):
        env.set_starting_player(starting_player)
        state = env.reset()
        while not env.done:
            player = env.current_player()
            agent = agents[int(player != "X")]  # get Agent
            available_actions = env.available_actions()
            action = agent.act(state, available_actions)
            state, reward, done, info = env.step(action)

        results.append(reward)

        # rotate starting player
        starting_player = swap_player(starting_player)

    scores = {"X": results.count(1), "O": results.count(-1)}
    scores["D"] = episodes - scores["X"] - scores["O"]

    return scores


@click.command(
    help="Train and benchmark AI Agents to learn TicTacToe, classic or quantum!"
)
@click.option(
    "--game",
    default="classic",
    type=click.Choice(games, case_sensitive=False),
)
@click.option("--board_size", default=3, type=click.IntRange(2, 6))
@click.option(
    "-p",
    "--player",
    default="ai",
    type=click.UNPROCESSED,
    callback=validate_player_spec,
    help="Player specified with the form: [human|simple|ai:key=value,other=option]",
)
@click.option(
    "--save_model_file", type=click.File("w"), help="Save trained model to disk"
)
@click.option("--episodes", default=1000, type=click.IntRange(1, 10000000))
def trainbench(game, board_size, player, episodes, save_model_file):
    player, player_options = player

    env = get_env_for_game(game, board_size)

    # decide players
    agents = [
        get_agent(player, "X", player_options, False, env),
        get_agent(player, "O", player_options, False, env),
    ]

    # Ensure at least one agent is trainable
    if not hasattr(agents[0], "train"):
        sys.exit("Error: Agent is not trainable!")

    do_train(env, agents, episodes)

    agents[1] = get_agent("simple", "O", None, False, env)
    scores = do_benchmark(env, agents)
    print("Benchmarking results (trained model X vs simple agent O):", scores)

    # write model & metadata to file
    data = {
        "game": game,
        "board_size": board_size,
        "date": datetime.utcnow().isoformat(),
        "options": player_options,
        "model": agents[0].export_model(),
    }
    json.dump(data, save_model_file, indent=2)
    print(f"Training complete, model saved to '{save_model_file.name}'")


@click.command(
    help="Benchmark AI Agents as they learn the game. Benchmark every 1000 training iterations"
)
@click.option(
    "--game",
    default="classic",
    type=click.Choice(games, case_sensitive=False),
)
@click.option("--board_size", default=3, type=click.IntRange(2, 6))
@click.option(
    "-p",
    "--player",
    default="ai",
    type=click.UNPROCESSED,
    callback=validate_player_spec,
    help="Player specified with the form: [human|simple|ai:key=value,other=option]",
)
@click.option(
    "--save_model_file", type=click.File("w"), help="Save trained model to disk"
)
@click.option(
    "--save_profile_file", type=click.File("w"), help="Save model profile data to disk"
)
@click.option("--episodes", default=1000, type=click.IntRange(1, 10000000))
def profiletraining(
    game, board_size, player, episodes, save_model_file, save_profile_file
):
    player, player_options = player

    env = get_env_for_game(game, board_size)
    interval = 1000
    step_count = int(episodes / interval)

    # decide players for self-play training
    shared_play_agents = [
        get_agent(player, "X", player_options, False, env),
        get_agent(player, "O", player_options, False, env),
    ]
    # to benchmark, pit one of the trained players against the simple agent
    benchmark_agents = [
        shared_play_agents[0],
        get_agent("simple", "O", None, False, env),
    ]

    # Ensure at least one agent is trainable
    if not hasattr(shared_play_agents[0], "train"):
        sys.exit("Error: Agent is not trainable!")

    profile = []
    print(
        f"Training model for {interval} iterations, then benchmarking. Repeating {step_count} times until completion..."
    )
    for i in range(step_count):
        print(f"Training step {i} of {step_count}")
        do_train(env, shared_play_agents, episodes=interval)
        scores = do_benchmark(env, benchmark_agents, episodes=1000)
        profile.append(scores | {"training_steps": i * interval})

    # write model & metadata to file
    data = {
        "game": game,
        "board_size": board_size,
        "date": datetime.utcnow().isoformat(),
        "options": player_options,
        "model": shared_play_agents[0].export_model(),
    }
    json.dump(data, save_model_file, indent=2)
    print(f"Training complete, model saved to '{save_model_file.name}'")

    # Write profile results to file
    profile_data = {
        "game": game,
        "board_size": board_size,
        "date": datetime.utcnow().isoformat(),
        "options": player_options,
        "profile": profile,
    }
    json.dump(profile_data, save_profile_file)
    print(f"Model training profile saved to '{save_profile_file.name}'")


cli.add_command(play)
cli.add_command(train)
cli.add_command(benchmark)
cli.add_command(trainbench)
cli.add_command(profiletraining)

if __name__ == "__main__":
    cli()
