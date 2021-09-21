#!/usr/bin/python3

from ray import tune
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.bayesopt import BayesOptSearch

from main import get_env_for_game, get_agent, do_benchmark, do_train

game = "classic"
board_size = 3
player = "quantumqlearning"
episodes_per_epoch = 2000


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

    for epoch in range(50):
        # Iterative training function
        do_train(env, agents, episodes_per_epoch)

        scores = do_benchmark(env, bench_agents, episodes=500)

        intermediate_score = scores["X"] / scores["O"]

        # Feed the score back back to Tune.
        tune.report(score=intermediate_score)


analysis = tune.run(
    training_function,
    config={
        "alpha": tune.uniform(0.5, 0.99),
        "k": tune.uniform(0.2, 0.99),
        "gamma": tune.uniform(0.1, 0.99),
    },
    metric="score",
    mode="max",
    search_alg=ConcurrencyLimiter(
        BayesOptSearch(random_search_steps=8), max_concurrent=4
    ),
    num_samples=40,
    stop={"training_iteration": 40},
)

print("Best config: ", analysis.get_best_config(metric="score", mode="max"))

# Get a dataframe for analyzing trial results.
df = analysis.results_df
print(df)
