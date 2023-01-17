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
    - [Mivia Audio Events Dataset](https://mivia.unisa.it/datasets/audio-analysis/mivia-audio-events/): `mivia_audio_events`
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
