import argparse
import os
import sys

import pandas as pd


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent, FuzzyQLAgent
from sumo_rl.exploration import EpsilonGreedy

from user.observation import CustomObservationFunction

if __name__ == "__main__":
    alpha = 0.1
    gamma = 0.99
    decay = 1
    runs = 1
    episodes = 1

    env = SumoEnvironment(
        net_file="user/nets/osm.net.xml.gz",
        route_file="user/nets/osm.passenger.trips.xml",  # trips can actually work as well
        use_gui=True,
        num_seconds=80000,
        min_green=5,
        delta_time=5,
        observation_class=CustomObservationFunction,   # Use custom observation function
    )

    for run in range(1, runs + 1):
        initial_states = env.reset()
        ql_agents = {
            ts: FuzzyQLAgent(
                starting_state=env.encode(initial_states[ts], ts),
                state_space=env.observation_space,
                action_space=env.action_space,
                alpha=alpha,
                gamma=gamma,
                fuzzy_dims= range( env.traffic_signals[ts].num_green_phases+1,  env.traffic_signals[ts].num_green_phases + 2*len(env.traffic_signals[ts].lanes)),
                exploration_strategy=EpsilonGreedy(initial_epsilon=0.05, min_epsilon=0.005, decay=decay),
            )
            for ts in env.ts_ids
        }

        for episode in range(1, episodes + 1):
            if episode != 1:
                initial_states = env.reset()
                for ts in initial_states.keys():
                    ql_agents[ts].state = env.encode(initial_states[ts], ts)

            infos = []
            done = {"__all__": False}
            while not done["__all__"]:
                actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

                s, r, done, info = env.step(action=actions)

                for agent_id in s.keys():
                    ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])

            env.save_csv(f"outputs/user/ql-osm_run{run}", episode)

    env.close()
