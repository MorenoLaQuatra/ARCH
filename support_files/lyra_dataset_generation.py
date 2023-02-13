import pandas as pd
import os

import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Lyra Dataset Generation')
parser.add_argument('--repo_path', type=str, default='/path/to/the/cloned/repo', help='Path to the cloned repo from github') # /data1/mlaquatra/datasets/audio_datasets/lyra-dataset/
parser.add_argument('--dataset_path', type=str, default='/path/to/the/dataset', help='Path to the final dataset (if the folder does not exist, it will be created)') # /data1/mlaquatra/datasets/audio_datasets/lyra-audio-clips/

args = parser.parse_args()

# check if dataset_path exists - if not ask to create it using a prompt
if not os.path.exists(args.dataset_path):
    print(f"Dataset path {args.dataset_path} does not exist. Do you want to create it? [y/n]")
    answer = input()
    if answer == 'y':
        os.mkdir(args.dataset_path)
    else:
        print("Exiting...")
        exit()

# Load the metadata - They are in data/raw.csv
metadata = pd.read_csv(os.path.join(args.repo_path, 'data', 'raw.tsv'), sep='\t')

# youtube-id column contains the youtube video id to be downloaded
# start-ts and end-ts columns contain the start and end timestamps of the audio clip to be extracted (seconds)
# id column contains the id of the audio clip - use it as the filename

def download_sample(
    youtube_id: str,
    start_ts: float,
    end_ts: float,
    output_path: str,
):  
    # check if the audio clip already exists
    if os.path.exists(output_path):
        return
    # execute the OS command to download the audio clip - wait until the download is complete
    os.system(f'yt-dlp -x --audio-format vorbis --audio-quality 5 --output "{output_path}" --postprocessor-args "-ss {start_ts} -to {end_ts}" https://www.youtube.com/watch?v={youtube_id}')

# download the audio clips in parallel
from joblib import Parallel, delayed
Parallel(n_jobs=8)(delayed(download_sample)(
    youtube_id=row['youtube-id'],
    start_ts=row['start-ts'],
    end_ts=row['end-ts'],
    output_path=os.path.join(args.dataset_path, f"{row['id']}.ogg")
) for index, row in tqdm(metadata.iterrows()))

# finished
print("Finished downloading the dataset")
