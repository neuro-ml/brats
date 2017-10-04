# Readme

This folder contains procedures to recreate BraTS 2017 submission of our team.

## Requirements

1. Python 3.6 and bash command `python` runs python3.6
2. GPU with CUDA support
3. Reqirements from [deep_pipe](https://github.com/neuro-ml/deep_pipe/blob/ef2bce0d81b95e1f5a22d3b013358d18f5d5fc96/requirements.txt)
4. Dataset BraTS 2017. We will consider that it exists and located in BRATS_PATH.
So, training part is located at BRATS_PATH/train, validation part is located at
BRATS_PATH/val and testing part is located in BRATS_PATH/test

## How to recreate

### Simple

Run `./recreate_brats2017 BRATS_PATH`

Predictions are located at

### All steps

1. `cd complex`
1. `./install_deep_pipe.sh`
1. `./prepare_data.sh`
2. `./build_configs.sh`
1. Run `prepare_brats2017.sh BRATS_PATH`