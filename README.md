# ClassificationSpeechNet

This is part of speech rate measure project.

This code can be used to train a multi class classification model to predict number of
vowels in an audio file.

## Installation instructions

* Python 3.6+
* torch
* argparse
* numpy
* soundfile
* librosa

* Download the code:
 ```
    git clone https://github.com/Jenny-Smolensky/ClassificationSpeechNet.git
 ```

## Data
Train/Validation/Test files should be in next the format:
```
├───0
│       100050.wav
│       100060.wav
│       100085.wav
├───1
│       100050.wav
│       100060.wav
│       100085.wav
├───2
│       100050.wav
│       100060.wav
│       100085.wav
....
Each folder indicated a number of vowels should contain the audio files with
 this vowels number.
In All three folders should have same amount of folders.
```
Create data using  libri speech files parser project (also contains sample files in suatable format)
[here](https://github.com/Jenny-Smolensky/libri-speech-files-parser)

## Parameters
number of classes - The number of different folders you have in train, validation and test
folders.

number of workers - depends on computer. 
more documention on [pytorch](https://pytorch.org/docs/stable/data.html)

## Trained model
We trained a model using libri-speech 100 hours speech data. 
you can find the trained model on : https://github.com/almog1/SpeechMeasureServer/tree/main/neural_net/trained/multiclass_model.pth

## Authors

**Almog Gueta** ,  **Jenny Smolensky** 

