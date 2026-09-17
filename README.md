# AT-DGNN

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/meeg-and-at-dgnn-advancing-eeg-emotion/eeg-emotion-recognition-on-meeg)](https://paperswithcode.com/sota/eeg-emotion-recognition-on-meeg?p=meeg-and-at-dgnn-advancing-eeg-emotion)

[BIBM2024] MEEG and AT-DGNN: Improving EEG Emotion Recognition with Music Introducing and Graph-based Learning
- Attention-Based Temporal Learner With Dynamical Graph Neural Network for EEG Emotion Recognition.

## Introduction 📖

The MEEG dataset,
capturing emotional responses to various musical stimuli across different valence and arousal levels,
enables an in-depth analysis of brainwave patterns within musical contexts.
We introduce the Attention-based Temporal Learner with Dynamic Graph Neural Network (AT-DGNN),
a novel framework for EEG emotion recognition.
By integrating an attention mechanism with a dynamic graph neural network (DGNN),
the AT-DGNN model captures complex local and global EEG dynamics,
demonstrating superior performance with accuracy of 83.74% in arousal and 86.01% in valence,
outperforming current state-of-the-art (SOTA) methods.

## Paper 📄

[MEEG and AT-DGNN: Improving EEG Emotion Recognition with Music Introducing and Graph-based Learning](https://ieeexplore.ieee.org/document/10821806)

## Network 🧠

![AT-DGNN](docs/assert/network.jpg)

The AT-DGNN model comprises two core modules: a feature extraction module (a) and a dynamic graph neural network learning module (b). The feature extraction module consists of a temporal learner, a multi-head attention mechanism, and a temporal convolution module. These components effectively leverage local features of EEG signals through a sliding window technique, thereby enhancing the model's capacity to dynamically extract complex temporal patterns in EEG signals. In the graph-based learning module, the model initially employs local filtering layers to segment and filter features from specific brain regions. Subsequently, the architecture employs three layers of stacked dynamic graph convolutions to capture complex interactions among different brain regions. This structure enhances the AT-DGNN's capacity for integrating temporal features effectively.

## Run 🏃

**The source code is totally compatible with DEAP dataset and MEEG dataset.** You can refer to the [run](docs/run.md) to run the code.

## Dataset 📊

If you are interested in the MEEG dataset, you can click [here](https://drive.google.com/drive/folders/1Tabw5sjpFiwy88yP-C-LnunNFrrre9AR?usp=drive_link) to download.

## Band Attention (this work) 🆕

An extension of AT-DGNN that makes the **frequency-band weighting explicit, sample-adaptive
and interpretable**. AT-DGNN's Tception temporal learner is a learnable filter bank followed by
a log-power layer, i.e. a *learned spectral decomposition*, but its frequency weighting is

1. **static** - fixed by the trained convolution kernels, identical for every sample;
2. **not aligned with canonical EEG bands** - each kernel is an arbitrary learned response;
3. **never re-weighted** - the three branches are simply concatenated.

The proposed module addresses all three points:

```
x -> fixed band split (delta/theta/alpha/beta/gamma, brick-wall masks)
  -> per-band log-power descriptor
  -> score  s_k = b_k (learnable static prior) + phi(d_k - mean d) (shared, sample-adaptive)
  -> beta = K * softmax(s),  x' = sum_k beta_k * x_k
  -> unchanged AT-DGNN backbone
```

Key properties (all verified, see `experiments/`):

| property | value |
|---|---|
| extra parameters | **+30** (0.0011 % of the 2 680 350 baseline parameters) |
| band decomposition | complementary FFT masks forming a partition of unity over 1-50 Hz; measured in-band energy fraction **100.0 %** per band (no leakage) |
| safety | with uniform weights the module reproduces the input, and `--band-attn none` is **bit-identical** to AT-DGNN (max abs diff 0.0) |
| interpretability | `beta` has 5 entries with clear band semantics, normalised to mean 1 |

```shell
# full module
python main.py --model 'AT-DGNN-BandAttn' --data-path ./example --dataset MEEG   --sampling-rate=1000 --target-rate=200 --trial-duration=59 --input-shape '1,32,800' --subjects=1

# ablations: static prior only / sample-adaptive modulation only
python main.py ... --model 'AT-DGNN-BandAttn' --band-attn static
python main.py ... --model 'AT-DGNN-BandAttn' --band-attn adaptive
```

### Controlled single-subject pilot

```shell
python experiments/pilot_bandattn.py --seed 3407 --folds 3 --epochs 40
python experiments/summarize_bandattn.py      # paired tables (baseline reused from archive)
```

### Cross-subject (LOSO) evaluation

```shell
# after preparing one sub<N>.hdf per subject (see docs/run.md)
python experiments/loso.py --variants "AT-DGNN" "AT-DGNN-BandAttn"   --data-folder data_eeg_MEEG_A --out experiments/results/loso_MEEG.json
python experiments/summarize_loso.py --pattern 'experiments/results/loso_MEEG.json'
```

### Paper draft

`docs/paper/ATDGNN-BandAttn-论文初稿.md`

## Models 📕

These models compared with AT-DGNN and unitized in the source code are listed in the [reference](docs/reference.md). 

- LGGNet
- EEGNet
- DeepConvNet
- ShallowConvNet
- TSception
- EEG-TCNet
- TCN-Fusion
- ATCNet
- DGCNN

## Visualization 📈

The visualization of the EEG signals can be found in the [visualization](docs/visualization.md).

## Citation 🖊️

If you find our work useful, please consider citing our paper:

```
@inproceedings{xiao2024meeg,
  title={MEEG and AT-DGNN: Improving EEG Emotion Recognition with Music Introducing and Graph-based Learning},
  author={Xiao, Minghao and Zhu, Zhengxi and Xie, Kang and Jiang, Bin},
  booktitle={2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  pages={4201--4208},
  year={2024},
  organization={IEEE Computer Society}
}
```

## Acknowledgement ✉️

The music for introducing the emotional state of the participants in the MEEG dataset is provided by [Rui Zhang Prof.](https://www.art.sdu.edu.cn/info/1499/14819.htm), Shandong University, Department of Music.

Some of the source code is originally from [LGGNet](https://github.com/yi-ding-cs/LGG). We appreciate the authors for their contribution.

This repository is derived from [AT-DGNN](https://github.com/xmh1011/AT-DGNN) (Apache License 2.0).
The band-attention module, the LOSO evaluation harness and the scripts in `experiments/` are
additions by the present author. The AT-DGNN baseline code is unchanged except for two minimal
fixes: `normalize_adjacency_matrix` used the module-level `DEVICE` constant (which crashes when a
GPU is present but the model is on CPU) and `train/train_model.py` relied on a star-import side
effect for `os`.

## Star ⭐️

If you find our code and dataset useful, we will be appreciate if you can give our repository a star.
