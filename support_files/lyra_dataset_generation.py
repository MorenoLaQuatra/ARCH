import pandas as pd
import numpy as np
import torchaudio

import argparse

parser = argparse.ArgumentParser(description='Lyra Dataset Generation')
parser.add_argument('--repo_path', type=str, default='/path/to/the/cloned/repo', help='Path to the cloned repo from github')
parser.add_argument('--dataset_path', type=str, default='/path/to/the/dataset', help='Path to the final dataset (if the folder does not exist, it will be created)')