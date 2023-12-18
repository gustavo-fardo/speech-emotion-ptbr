## Speech Emotion in Portuguese - BR 
This repository contains the develompent of classification models for recognizing emotions in speech based on prosody (intonation, rythm, stress, etc.). The data used is the [emoUERJ](https://zenodo.org/records/5427549) open dataset. The Jupyter Notebooks detail several methods of feature extraction, tested on a Support Vector Machine classifier and a custom Deep Learning model built with Keras.

## Requirements
- Python 3.9
- Dataset [emoUERJ](https://zenodo.org/records/5427549)

## Installation
Repository installation
``` 
git clone https://github.com/gustavo-fardo/projeto-embarcado-ima
cd ./projeto-embarcado-ima
```
Create a python virtual environment (recommended):
```
sudo apt install python3.9
sudo apt install virtualenv
python3.9 -m virtualenv .venv --python=$(which python3.9)
```
**OBS**: every time you open a new terminal, activate the virtual environment with the command:
```
source .venv/bin/activate
```
Deactivate it with:
```
deactivate
```

*To reproduce results, dowloading [emoUERJ](https://zenodo.org/records/5427549) is needed, and then put it inside /datasets folder*

## Data Augmentation
The data augmentation methods, using the [Audiomentations](https://github.com/iver56/audiomentations) library, were used to triple the size of the dataset, and are the following:

- Gaussian Noise with random amplification between 0.001 and 0.01
- Time Stretch between 0.8 and 1.24 times
- Pitch Shift with random semitone variation of -2 to 2

The data augmentation process is detailed in feat_extract.ipynb and feat_extract2.ipynb.

## Feature Extraction
The feature extraction methods tested on [emoUERJ](https://zenodo.org/records/5427549) are documented in the feat_extract.ipynb and feat_extract2.ipynb, with the following:

- [TSFEL](https://github.com/fraunhoferportugal/tsfel) library (too slow, not tested with the models)
- Praat e Parselmouth, based on https://github.com/uzaymacar/simple-speech-features
- [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis) library
- Mel-Frequency Cepstrum Coefficients (MFCC) with [Librosa library]()
- Mel Spectrogram with [Librosa library]()

## Authors
- [Daniel Augusto Pires de Castro](https://github.com/daniapc).
- [Gustavo Fardo Armênio](https://github.com/gustavo-fardo).

