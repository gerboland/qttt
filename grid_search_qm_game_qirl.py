#!/usr/bin/python3

from ray import tune
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.bayesopt import BayesOptSearch

from main import get_env_for_game, get_agent, do_benchmark, do_train

game = "qubit"
board_size = 3
player = "qiql"
episodes_per_epoch = 2500


def training_function(config):
    # Hyperparameters
    alpha, k, gamma = config["alpha"], config["k"], config["gamma"]

    player_options = {
        "alpha": alpha,
        "k": k,
        "gamma": gamma,
    }

    env = get_env_for_game(game, board_size)

    # decide players
    agents = [
        get_agent(player, "X", player_options, False, env),
        get_agent(player, "O", player_options, False, env),
    ]

    bench_agents = [agents[0], get_agent("simple", "O", None, False, env)]

    for epoch in range(40):
        # Iterative training function
        do_train(env, agents, episodes_per_epoch)

        scores = do_benchmark(env, bench_agents, episodes=100)

        intermediate_score = scores["X"] / scores["O"]

        # Feed the score back back to Tune.
        tune.report(score=intermediate_score)


analysis = tune.run(
    training_function,
    config={
        "alpha": tune.uniform(0.3, 0.8),
        "k": tune.uniform(0.4, 0.8),
        "gamma": tune.uniform(0.6, 0.99),
    },
    metric="score",
    mode="max",
    search_alg=ConcurrencyLimiter(
        BayesOptSearch(random_search_steps=8), max_concurrent=8
    ),
    num_samples=20,
    stop={"training_iteration": 20},
)

print("Best config: ", analysis.get_best_config(metric="score", mode="max"))

# Get a dataframe for analyzing trial results.
df = analysis.results_df
print(df)
