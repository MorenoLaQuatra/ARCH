# ARCH: Audio Representations benCHmark

ARCH is a benchmark for audio representations. Its goal is to provide an unified framework for the evaluation of audio representations.
The benchmark is intended to be **easy to use** and to **allow the comparison of different audio representations**.

The main features of ARCH are:
- **Easy to use**: the benchmark is designed to be easy to use.
- **Comprehensive evaluation**: the benchmark provides a comprehensive evaluation of audio representations. It aims at providing a unified framework for the evaluation of audio representations on a wide range of tasks.
- **Standardization**: the benchmark wants to standardize the evaluation of audio representations. The pletora of ARL models and datasets makes it difficult to compare them. The benchmark aims at providing a standard way to evaluate audio representations.

## Installation

TODO

## Datasets and tasks

The benchmark includes the following tasks and, for each task, the corresponding datasets:

**Audio Events**

- Classification
    - [ESC-50](https://github.com/karolpiczak/ESC-50): `esc50`
- Sequence tagging
    - [Mivia Audio Events Dataset](https://mivigdowna.unisa.it/datasets/audio-analysis/mivia-audio-events/): `mivia_audio_events`
    - [Mivia road Audio Events Dataset](https://mivia.unisa.it/datasets/audio-analysis/mivia-road-audio-events-data-set/): `mivia_road_audio_events`


**Music**

- Classification
    - [FMA-small](https://github.com/mdeff/fma): `fma_small`
- Sequence tagging
    - [MUSDB18](https://sigsep.github.io/datasets/musdb.html): `musdb18`


**Speech**
- Classification
    - [RAVDESS](https://zenodo.org/record/1188976): `ravdess`

- Sequence tagging
    - [MUSAN](https://www.openslr.org/17/): `musan`


## Datasets download

### ESC-50
```bash
wget https://github.com/karoldvl/ESC-50/archive/master.zip
unzip master.zip
rm master.zip
mv ESC-50-master esc50
```

### FMA-small
It needs to have 7z installed given a known error with unzip (`error: not enough memory for bomb detection`). If you don't have it, you can install it with:
```bash
sudo apt-get install p7zip-full
```

Then, download and unzip the dataset:
```bash
mkdir fma_small
cd fma_small
wget https://os.unil.cloud.switch.ch/fma/fma_small.zip
wget https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
7z x fma_small.zip
7z x fma_metadata.zip
rm fma_small.zip
rm fma_metadata.zip
```

### RAVDESS
```bash
wget https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip
mkdir ravdess
unzip Audio_Speech_Actors_01-24.zip -d ravdess
rm Audio_Speech_Actors_01-24.zip


mv Audio_Speech_Actors_01-24 ravdess
```

### MUSAN - NOT USED AT THE MOMENT

```bash
wget https://www.openslr.org/resources/17/musan.tar.gz
tar -xvzf musan.tar.gz
rm musan.tar.gz
```

### MUSDB18

```bash
wget https://zenodo.org/record/3338373/files/musdb18hq.zip
unzip musdb18hq.zip
rm musdb18hq.zip
```

### Mivia Audio Events Dataset

To download the dataset, you need to register on the [Mivia](https://mivia.unisa.it/) website. Both for the audio events dataset and the road audio events dataset, you can download the dataset for free if used for research purposes.

```bash
# download the dataset
unzip mivia_audio_events.zip
rm mivia_audio_events.zip
```

# Mivia Road Audio Events Dataset

Similarly to the audio events dataset, you need to register on the [Mivia](https://mivia.unisa.it/) website to download the dataset.

```bash
# download the dataset
unzip MIVIA_ROAD_DB1.zip
rm MIVIA_ROAD_DB1.zip
```

# MIR-1K

```bash
wget http://mirlab.org/dataset/public/MIR-1K.zip
unzip MIR-1K.zip
rm MIR-1K.zip
```

# Jamendo

```bash
wget https://zenodo.org/record/2585988/files/jamando.zip
unzip jamando.zip
rm jamando.zip
```


---

# UrbanSound8K

```bash
wget https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz
tar -xvzf UrbanSound8K.tar.gz
rm UrbanSound8K.tar.gz
```

# Lyra Dataset (TODO)

```bash
git clone https://github.com/pxaris/lyra-dataset.git
wget script
```

# AudioMNIST

```bash
git clone https://github.com/soerenab/AudioMNIST.git
```

# MagnaTagATune

The split of the dataset is the same as the one used in [Musicnn](https://github.com/jordipons/musicnn-training/tree/master/data/index/mtt).

```bash

mkdir magnatagatune
wget https://mirg.city.ac.uk/datasets/magnatagatune/clip_info_final.csv
wget https://mirg.city.ac.uk/datasets/magnatagatune/mp3.zip.001
wget https://mirg.city.ac.uk/datasets/magnatagatune/mp3.zip.002
wget https://mirg.city.ac.uk/datasets/magnatagatune/mp3.zip.003
wget https://mirg.city.ac.uk/datasets/magnatagatune/annotations_final.csv
cat mp3.zip.* > mp3.zip
unzip mp3.zip
rm mp3.zip
rm mp3.zip.*

wget https://raw.githubusercontent.com/jordipons/musicnn-training/master/data/index/mtt/test_gt_mtt.tsv # download the test split
wget https://raw.githubusercontent.com/jordipons/musicnn-training/master/data/index/mtt/train_gt_mtt.tsv # download the train split
wget https://raw.githubusercontent.com/jordipons/musicnn-training/master/data/index/mtt/val_gt_mtt.tsv # download the validation split

```

# IRMAS

```bash
mkdir irmas
cd irmas

wget https://zenodo.org/record/1290750/files/IRMAS-TrainingData.zip
wget https://zenodo.org/record/1290750/files/IRMAS-TestingData-Part1.zip
wget https://zenodo.org/record/1290750/files/IRMAS-TestingData-Part2.zip
wget https://zenodo.org/record/1290750/files/IRMAS-TestingData-Part3.zip

unzip IRMAS-TrainingData.zip
unzip IRMAS-TestingData-Part1.zip
unzip IRMAS-TestingData-Part2.zip
unzip IRMAS-TestingData-Part3.zip

rm IRMAS-TrainingData.zip
rm IRMAS-TestingData-Part1.zip
rm IRMAS-TestingData-Part2.zip
rm IRMAS-TestingData-Part3.zip
```

# FSD50K

```bash
mkdir fsd50k
cd fsd50k
wget https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z01
wget https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z02
wget https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z03
wget https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z04
wget https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z05
wget https://zenodo.org/record/4060432/files/FSD50K.dev_audio.zip

wget https://zenodo.org/record/4060432/files/FSD50K.eval_audio.z01
wget https://zenodo.org/record/4060432/files/FSD50K.eval_audio.zip

wget https://zenodo.org/record/4060432/files/FSD50K.ground_truth.zip
wget https://zenodo.org/record/4060432/files/FSD50K.metadata.zip


7z x FSD50K.dev_audio.zip
rm FSD50K.dev_audio.z*

7z x FSD50K.eval_audio.zip
rm FSD50K.eval_audio.z*

unzip FSD50K.ground_truth.zip
unzip FSD50K.metadata.zip

rm FSD50K.ground_truth.zip
rm FSD50K.metadata.zip
```

# VIVAE

```bash
mkdir vivae
cd vivae
wget https://zenodo.org/record/4066235/files/VIVAE.zip
unzip VIVAE.zip
rm VIVAE.zip
```

# SLURP

```bash
mkdir slurp
cd slurp
wget https://zenodo.org/record/4274930/files/slurp_real.tar.gz
wget https://zenodo.org/record/4274930/files/slurp_synth.tar.gz
tar -xvzf slurp_real.tar.gz
tar -xvzf slurp_synth.tar.gz
rm slurp_real.tar.gz
rm slurp_synth.tar.gz
wget https://raw.githubusercontent.com/pswietojanski/slurp/master/dataset/slurp/devel.jsonl
wget https://raw.githubusercontent.com/pswietojanski/slurp/master/dataset/slurp/test.jsonl
wget https://raw.githubusercontent.com/pswietojanski/slurp/master/dataset/slurp/train.jsonl
```

# Ballroom

```bash
mkdir ballroom
cd ballroom
wget http://mtg.upf.edu/ismir2004/contest/tempoContest/data1.tar.gz
wget http://mtg.upf.edu/ismir2004/contest/tempoContest/data2.tar.gz

tar -xvzf data1.tar.gz
tar -xvzf data2.tar.gz

rm data1.tar.gz
rm data2.tar.gz
```

# Clotho

```bash
mkdir clotho
cd clotho
wget https://zenodo.org/record/3490684/files/clotho_audio_development.7z
wget https://zenodo.org/record/3490684/files/clotho_audio_evaluation.7z
wget https://zenodo.org/record/3490684/files/clotho_captions_development.csv
wget https://zenodo.org/record/3490684/files/clotho_captions_evaluation.csv
wget https://zenodo.org/record/3490684/files/clotho_metadata_development.csv
wget https://zenodo.org/record/3490684/files/clotho_metadata_evaluation.csv

7z x clotho_audio_development.7z
7z x clotho_audio_evaluation.7z

rm clotho_audio_development.7z
rm clotho_audio_evaluation.7z
```

# Emotify

```bash
mkdir emotify
cd emotify
wget http://www2.projects.science.uu.nl/memotion/emotifydata/emotifymusic.zip
wget http://www2.projects.science.uu.nl/memotion/emotifydata/data.csv
unzip emotifymusic.zip
rm emotifymusic.zip
```

# MedleyDB

```bash
mkdir medleydb
cd medleydb
wget https://zenodo.org/record/3464194/files/Medley-solos-DB.tar.gz
wget https://zenodo.org/record/2582103/files/Medley-solos-DB_metadata.csv
mkdir audio
cd audio
mv ../Medley-solos-DB.tar.gz .
tar -xvzf Medley-solos-DB.tar.gz
rm Medley-solos-DB.tar.gz
```

# AccentDB

```bash
mkdir accentdb
cd accentdb
gdown 1a5cN4GwzsngrpYP230hzM58I8BhtB8et
# accentdb_core.tar.gz
tar -xvzf accentdb_core.tar.gz
rm accentdb_core.tar.gz
```

# EMOVO

```bash
mkdir emovo
cd emovo
gdown 1SUtaKeA-LYnKaD3qv87Y5wYgihJiNJAo
unzip emovo.zip
rm emovo.zip
```




## Usage

TODO

## Contributing

TODO

## License

TODO

## Authors

TODO

## Acknowledgments

TODO

## References

TODO

## Cite

TODO
