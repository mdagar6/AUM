from Pipeline import pipeline
import argparse
import configparser
import random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if  __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read("config.ini")

    # section identify "section" in config.ini
    parser = argparse.ArgumentParser()
    parser.add_argument('--section', type=str)
    args = parser.parse_args()
    section = args.section
    config = config[section]

    #Run pipeline
    random_seed = config["random_seed"]
    random_seed = list(map(int, random_seed.split(",")))

    synthetic_noise = config["synthetic_noise"]
    synthetic_noise = list(map(float, synthetic_noise.split(",")))

    for seed, noise in zip(random_seed, synthetic_noise):
        print("For: seed -", seed, "noise -", noise)
        set_seed(seed)
        pipeline.run(config,seed, noise)
