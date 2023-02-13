import pandas as pd
import os
import glob

import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Lyra Dataset Statistics')
parser.add_argument('--repo_path', type=str, default='/path/to/the/cloned/repo', help='Path to the cloned repo from github') # --repo_path /data1/mlaquatra/datasets/audio_datasets/lyra-dataset/
parser.add_argument('--dataset_path', type=str, default='/path/to/the/dataset', help='Path to the final dataset (if the folder does not exist, it will be created)') # --dataset_path /data1/mlaquatra/datasets/audio_datasets/lyra-audio-clips/
args = parser.parse_args()

# find all ogg files in the dataset_path
ogg_files = glob.glob(os.path.join(args.dataset_path, '*.ogg'))

# load the metadata from repo_path + data/raw.tsv
metadata = pd.read_csv(os.path.join(args.repo_path, 'data', 'raw.tsv'), sep='\t')

# compare the number of ogg files with the number of rows in the metadata
print(f"Number of ogg files: {len(ogg_files)}")
print(f"Number of rows in metadata: {len(metadata)}")
print(f"Missing files: {len(metadata) - len(ogg_files)}")

# for each row in the metadata that has a corresponding ogg file - get the list of labels using the instruments column and compute some statistics
# the instruments column contains a list of instruments separated by "|"

list_instruments_labels = []
for index, row in metadata.iterrows():
    if os.path.exists(os.path.join(args.dataset_path, f"{row['id']}.ogg")):
        instruments = row['instruments'].split('|')
        list_instruments_labels.append(instruments)

# compute the number of unique labels
unique_labels = set([item for sublist in list_instruments_labels for item in sublist])

# compute the number of samples per label
samples_per_label = {}
for label in unique_labels:
    samples_per_label[label] = sum([label in sublist for sublist in list_instruments_labels])

# print the results
print(f"\n\nNumber of unique labels [INSTRUMENTS]: {len(unique_labels)}")
# sort the labels by number of samples
samples_per_label = {k: v for k, v in sorted(samples_per_label.items(), key=lambda item: item[1], reverse=True)}
for label, count in samples_per_label.items():
    print(f"{label}: {count}")

# the column genres contains a list of genres separated by "|"
list_genres_labels = []
for index, row in metadata.iterrows():
    if os.path.exists(os.path.join(args.dataset_path, f"{row['id']}.ogg")):
        genres = row['genres'].split('|')
        # take only the first genre
        genres = [genres[0]]
        list_genres_labels.append(genres)

# compute the number of unique labels
unique_labels = set([item for sublist in list_genres_labels for item in sublist])

# compute the number of samples per label
samples_per_label = {}
for label in unique_labels:
    samples_per_label[label] = sum([label in sublist for sublist in list_genres_labels])

# print the results
print(f"\n\nNumber of unique labels [GENRE]: {len(unique_labels)}")
samples_per_label = {k: v for k, v in sorted(samples_per_label.items(), key=lambda item: item[1], reverse=True)}
for label, count in samples_per_label.items():
    print(f"{label}: {count}")
