# Singing-Voice-Separation

A minimal project for singing voice separation. Every component is as simple as possible:

* The network is a traditional 3-layer-RNN with GRU cell. Each layer has 256 neurons only. The model file is only 12.5MB.

* It uses L2 loss and momentum optimizer, taking around 1600 iters to converge, which needs less than 4 hours on a 1080Ti.

* The dataset is [MIR-1k dataset](http://mirlab.org/dataset/public/), which is only around 650MB containing 1000 music clips.

Yet, it's quite powerful for its task (well, only on this dataset). Just check the demo.

## Usage

### Off-the-shell

1. Download the pre-trained model from [here](https://drive.google.com/file/d/1at1mEuPb92eJBHoOXFWnXGdSkktQ8Avu/view?usp=sharing). Extract it at the root directory of the project - make sure you have some files under `./log/baseline`.

2. Run `python eval.py`.

About how to use this tool, check `eval.process_single_example`. NOTE: the input file must be in `.wav` format with 16000 sample rate. You can easily convert a file into this using ffmpeg.

### Train

Why bother?

If you insist to train by yourself, you'll need to:

1. Download the dataset to `./data`.

2. Use `data.train_test_split` to split train and test set.

3. Use `train.py`.

## Dependencies

- librosa
- numpy
- tensorflow
- mir_eval

## Evaluation

| Source | GNDSR | GSIR  | GSAR |
| :--: | :---: | :---: | :--: |
| Accompanies | 7.78  | 10.68 | 8.11 |
| Vocal | 7.64  | 10.36 | 8.06 |

