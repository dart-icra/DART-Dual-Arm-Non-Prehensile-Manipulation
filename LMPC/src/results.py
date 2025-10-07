import numpy as np
import matplotlib.pyplot as plt

from analyitics import Logger
import os
import argparse as ap

parser = ap.ArgumentParser(description="")
parser.add_argument("--env", type=str, default="sphere_0x2_0x1", help="environment name")
parser.add_argument("--train", action="store_true", help="train or test")
parser.add_argument("--tag",type=str)
args = parser.parse_args()

datalogger = Logger(logdir="./data/logs")

#! =======================================================================
#! CHANGE THE PATH BELOW IN THE FORMAT <METRIC>_<OBJECT>_<MASS>_<FRICTION>
#! ALSO GO THROUGH WORLD.XML, TO UPDATE FRICTION IN 2 DIFFERENT PLACES
#! =======================================================================

phase = "train" if args.train else "test"
path = f"{args.tag}_test/{args.env}"

metrics = ["pos_error","u_cmd","timestep"]  # Add more metrics if needed

for metric in metrics:
    episodic_data = datalogger.load(id=metric, fpath=f"./data/logs/{path}.npy")

    num_episodes = len(episodic_data)
    if num_episodes == 0:
        raise RuntimeError(f"No data found for metric '{metric}' in {path}.npy")

    os.makedirs(f"./data/plots/{path}", exist_ok=True)

    for i, episode in enumerate(episodic_data):
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(episode) 
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"Episode {i+1}")
        fig.savefig(f"./data/plots/{path}/episode_{i+1}_{metric}.png")
        plt.close(fig)

    lowest_vals = [np.min(ep) for ep in episodic_data]
    avg_vals = [np.mean(ep) for ep in episodic_data]
    episodes = np.arange(1, num_episodes + 1)

    plt.figure(figsize=(10, 4))
    plt.plot(episodes, lowest_vals, marker='o')
    plt.xlabel("Episode")
    plt.ylabel(f"Lowest {metric.replace('_', ' ').title()}")
    plt.title(f"Lowest {metric.replace('_', ' ').title()} per Episode")
    plt.grid(True)
    plt.savefig(f"./data/plots/{path}/lowest_{metric}_per_episode.png")
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(episodes, avg_vals, marker='o', color='orange')
    plt.xlabel("Episode")
    plt.ylabel(f"Average {metric.replace('_', ' ').title()}")
    plt.title(f"Average {metric.replace('_', ' ').title()} per Episode")
    plt.grid(True)
    plt.savefig(f"./data/plots/{path}/avg_{metric}_per_episode.png")
    plt.close()
