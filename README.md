# ARCH: Audio Representations benCHmark

<p align="center">
  <img src="resources/arch_logo.png" alt="logo" width="400">
</p>


This repository contains the code for the ARCH benchmark. It is intended to be used to evaluate audio representations on a wide range of datasets and tasks.
The benchmark is intended to be **easy to use** and to **allow the comparison of different audio representations**.

The main features of ARCH are:
- **Plug and play**: the benchmark is designed to be easy to use. It provides a unified interface to load the datasets and to evaluate audio representations.
- **Extensibility**: the benchmark is designed to be easy to extend. It is possible to add new datasets and tasks as well as new models to evaluate its audio representations.
- **Standardization**: the benchmark wants to standardize the evaluation of audio representations. The pletora of ARL models and datasets makes it difficult to compare them. The benchmark aims at providing a standard way to evaluate audio representations.

## Installation

ARCH can be installed by just cloning the repository and installing it with pip:

```bash
git clone https://github.com/MorenoLaQuatra/ARCH.git
cd ARCH
pip install -e .
```

## Usage

The benchmark can be used by importing the `arch` module. The file `evaluate_hf_models.py` contains an example of how to use the benchmark. It contains the following parameters that can be used to configure the benchmark:

- `model`: the name of the model to evaluate. It can be any model from the [HuggingFace model hub](https://huggingface.co/models) or a local model exposing the same interface.
- `device`: the device to use for the evaluation. It can be `cpu` or `cuda`.
- `max_epochs`: the maximum number of epochs to train the linear classifier.
- `verbose`: if `True`, it prints the results of the evaluation and other information on the standard output.
- `tsv_logging_file`: the file where to save the results of the evaluation in TSV format.
- `n_iters`: the number of times to repeat the evaluation, it can be used to compute the average of multiple runs and their standard deviation.
- `data_config_file`: the file containing the configuration of the datasets to use for the evaluation (you can find it at `configs/data_config.json`)
- `enabled_datasets`: the list of datasets to use for the evaluation. It can be any of the following: `esc50`, `us8k`, `fsd50k`, `vivae`, `fma_small`, `magna_tag_a_tune`, `irmas`, `medleydb`, `ravdess`, `audio_mnist`, `slurp`, `emovo`.



# Datasets and tasks

The benchmark includes multiple datasets and, at the moment, only classification tasks. The following table contains the list of the datasets and tasks currently supported by the benchmark.

| Dataset | Task | Type | Reference | Version |
| --- | --- | --- | --- | --- |
| [ESC-50](https://github.com/karolpiczak/ESC-50) | Single-label classification | Sound events | [ESC: Dataset for Environmental Sound Classification](http://dx.doi.org/10.1145/2733373.2806390) | Version 1 |
| [US8K](https://urbansounddataset.weebly.com/urbansound8k.html) | Single-label classification | Sound events | [A Dataset and Taxonomy for Urban Sound Research](https://doi.org/10.1145/2647868.2655045) | Version 1 |
| [FSD50K](https://zenodo.org/record/4060432) | Single-label classification | Sound events | [FSD50K: An Open Dataset of Human-Labeled Sound Events](https://doi.org/10.1109/TASLP.2021.3133208) | Version 1 |
| [VIVAE](https://zenodo.org/record/4066235) | Single-label classification | Sound events | [The Variably Intense Vocalizations of Affect and Emotion (VIVAE) corpus prompts new perspective on nonspeech perception](http://dx.doi.org/10.1037/emo0001048) | Version 1 |
|  |  |  |  |
| [FMA-small](https://github.com/mdeff/fma) | Single-label classification | Music | [FMA: A Dataset For Music Analysis](https://arxiv.org/abs/1612.01840) | Version 1 |
| [MagnaTagATune](https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset) | Multi-label classification | Music | [Evaluation of algorithms using games: the case of music annotation](https://ismir2009.ismir.net/proceedings/OS5-5.pdf) | Version 1 |
| [IRMAS](https://zenodo.org/record/1290750) | Multi-label classification | Music | [A Comparison of Sound Segregation Techniques for Predominant Instrument Recognition in Musical Audio Signals](https://ismir2012.ismir.net/event/papers/559_ISMIR_2012.pdf) | Version 1 |
| [Medley-solos-DB](https://medleydb.weebly.com/) | Single-label classification | Music | [Deep convolutional networks on the pitch spiral for musical instrument recognition](https://archives.ismir.net/ismir2016/paper/000093.pdf) | Version 1 |
|  |  |  |  |
| [RAVDESS](https://zenodo.org/record/1188976) | Single-label classification | Speech | [The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English](https://doi.org/10.1371/journal.pone.0196391) | Version 1 |
| [AudioMNIST](https://github.com/soerenab/AudioMNIST) | Single-label classification | Speech | [Interpreting and Explaining Deep Neural Networks for Classification of Audio Signals](https://arxiv.org/abs/1807.03418) | Version 1 |
| [SLURP](https://zenodo.org/record/4274930) | Single-label classification | Speech | [SLURP: A Spoken Language Understanding Resource Package](https://aclanthology.org/2020.emnlp-main.588/) | Version 1 |
| [EMOVO](http://voice.fub.it/activities/corpora/emovo/index.html) | Single-label classification | Speech | [EMOVO: A Dataset for Emotion Recognition in Spontaneous Speech](https://aclanthology.org/L14-1478/) | Version 1 |


**Version 1**: 2022-02-26 - The first released version of the benchmark. The table above indicate which dataset is included for each version of the benchmark.

The instructions to download the datasets are available on the [data_download/README.md](data_download/README.md) file.

Detailed information and results of the first version of the benchmark are available on this [space](https://huggingface.co/spaces/ALM/ARCH). The results include both the numbers reported in the paper and the specific versions of the models evaluated.


## Usage

TODO

## Contributing

TODO

## License


<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

## Authors

[Moreno La Quatra](mlaquatra.me)
[Alkis Koudounas](https://koudounasalkis.github.io/)
[Lorenzo Vaiani]()

## Acknowledgments

This work could not have been possible without the support of the authors of the datasets and the models used in the benchmark. We would like to thank them for their work and for making their datasets and models publicly available.

## References

The table above contains the link and references of the **datasets** used in the benchmark, if you use them in your work, please cite them accordingly. 

The specific **models** evaluated for each version of the benchmark are reported in the [results](https://huggingface.co/spaces/ALM/ARCH) page, if you use them in your work, please cite them accordingly.

If you use the benchmark in your work, please cite the following paper:

**Version 1**:
```bibtex
@article{...COMING SOON...,
  title={ARCH: AUDIO REPRESENTATION BENCHMARK FOR CLASSIFICATION OF MUSIC,
SPEECH, AND SOUND EVENTS},
  author={La Quatra, Moreno and Koudounas, Alkis and Vaiani, Lorenzo and Baralis, Elena and Cagliero, Luca},
  journal={},
  year={2023}
}
```
