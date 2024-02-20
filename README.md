# VAE-OMAX (2021)
[SOMax](https://forum.ircam.fr/projects/detail/somax-2/) is a software developed at IRCAM by G. Assayag, M. Chemillier and G. Bloch aiming to generate improvisation based on a given material. The main idea
is to go navigate a musical corpus in a non linear way, jumping from one chord/note
to another that can be much further. This is done by "traveling" in a latent space of musical chords. In SOMax 2, this is made using [self-organizing maps](https://en.wikipedia.org/wiki/Self-organizing_map) (hence the SOM in SOMax). During my internship at IRCAM, I explored other ways to build this latent space. This repository focuses on [Variational Autoencoders](https://en.wikipedia.org/wiki/Variational_autoencoder), and contains the code used to train and evaluate the VAE's latent space.
## Description

This project contains the Python code I used to conduct my research during my internship at IRCAM in 2021. The code is divided into several scripts:
- `createDataset.py`: This script is used to create a dataset from raw MIDI files.
- `src/models.py`: This script defines the architecture of the VAE.
- `train.py`: This script is responsible for training the VAE.
- `evaluation.py`: This script contains functions to evaluate the performance of the VAE.
- `src/utils.py`: various utilitary functions

## Requirements

This project requires the following Python packages:

- ``numpy``
- ``pandas``
- ``tensorflow``
- ``keras``
- ``music21``
## Usage
```sh
python createDataset.py
python train.py [--epochs EPOCHS] [--coldstart COLDSTART] [--beta BETA] [--batchsize BATCHSIZE]
python evaluation.py
```

