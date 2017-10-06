# Readme

This folder contains procedures to recreate BraTS 2017 submission of our team.

There are two ways to run computations:
 - install the necessary packages in your OS;
 - use a Docker containter with all necessary software.

## Requirements for both ways to run
1. GPU with CUDA support at least 12 GB memory. Probably works with 6GB.
2. Dowloaded dataset BraTS 2017 (including validation and test sets). 
Assuming that BRATS_PATH is a path to the dataset, we require the following structure
 - the training sample is located at BRATS_PATH/train, 
 - the validation sample is located at BRATS_PATH/val and 
 - the test sample is located in BRATS_PATH/test.

## Manual installation 

### Requirements

1. Python 3.6 and bash command `python` runs python3.6
2. The requirements from [deep_pipe](https://github.com/neuro-ml/deep_pipe/blob/87c1d6814cd2172375288deb17c077f17678f76d/requirements.txt)

### How to recreate

Run `./recreate_brats2017 BRATS_PATH` 

## Docker

### Requirements

1. nvidia-docker

### How to recreate

Run `./dockerized_recreate_brats2017 BRATS_PATH`

## The results

Predictions are located in folders `val_submission` and `test_submission`

There is `clean.sh` script that clears all results.
