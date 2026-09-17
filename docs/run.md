# Run

## Environment

- Python 3.9

```shell
conda create -n atdgnn python=3.9
```

```shell
conda activate atdgnn
```

## Install Requirements

```shell
pip3 install -r requirements.txt
```

## Run

```shell
python main.py
```

You can set the parameters in `main.py` to run the code. You can also use flags to set the parameters. The parameters used in our paper are listed in the [params](./params.md) file.

```shell
python main.py --label 'V'
```

### DEAP

If you want to run the code with the DEAP dataset, you must set these parameters as follows:

```shell
python main.py --dataset 'DEAP' --sampling-rate=128 --target-rate=128 --trial-duration=63 --input-shape '1, 32, 512'
```

For other parameters, you can refer to the [params](./params.md) file.

### MEEG

If you want to run the code with the MEEG dataset, you must set these parameters as follows:

```shell
python main.py --dataset 'MEEG' --sampling-rate=1000 --target-rate=200 --trial-duration=59 --input-shape '1, 32, 800'
```

For other parameters, you can refer to the [params](./params.md) file.

## Example

- Clone our code.
  ```shell
  git clone https://github.com/xmh1011/AT-DGNN.git
  cd AT-DGNN
  ```
- Unzip the data file.
  ```shell
  cd example
  tar -xvzf s01.tar.gz
  tar -xvzf sample_1.tar.gz
  ```

`example/sample_1.dat` is subject 1 of MEEG dataset. `example/s01.dat` is subject 1 of DEAP dataset.

After unzipping the sample data, you can run the code as follows.

### MEEG

`example/sample_1.dat` is subject 1 of MEEG dataset.

You can run as follows:

```shell
python main.py --data-path './example' --dataset 'MEEG' --sampling-rate=1000 --target-rate=200 --trial-duration=59 --input-shape '1,32,800' --subjects=1 --model 'AT-DGNN'
```

### DEAP 

`example/s01.dat` is subject 1 of DEAP dataset.

You can run as follows:

```shell
python main.py --data-path './example' --dataset 'DEAP' --sampling-rate=128 --target-rate=128 --trial-duration=63 --input-shape '1,32,512' --subjects=1 --model 'AT-DGNN'
```

## Reproduce

If you want to reproduce the results in our paper, you can download the dataset. After downloading the dataset, you can run the code as mentioned above.

## Band Attention

The `AT-DGNN-BandAttn` model adds a sample-adaptive weighting over the five canonical EEG
bands in front of the unchanged AT-DGNN backbone (see the README for the design).

```shell
# full module (static prior + sample-adaptive modulation)
python main.py --model 'AT-DGNN-BandAttn' --data-path ./example --dataset MEEG   --sampling-rate=1000 --target-rate=200 --trial-duration=59 --input-shape '1,32,800' --subjects=1

# ablations
python main.py ... --model 'AT-DGNN-BandAttn' --band-attn static     # learnable band prior only
python main.py ... --model 'AT-DGNN-BandAttn' --band-attn adaptive   # sample-adaptive only
python main.py ... --model 'AT-DGNN-BandAttn' --band-attn none       # == AT-DGNN baseline

# FIR band splitting instead of brick-wall FFT masks (for comparison)
python main.py ... --model 'AT-DGNN-BandAttn' --band-kind fir --band-numtaps 129
```

### Band-attention pilot (single subject)

```shell
python experiments/pilot_bandattn.py --seed 3407 --folds 3 --epochs 40
python experiments/summarize_bandattn.py
```

The pilot trains the band-attention variants; the baseline is reused from the archived runs
because `--band-attn none` was verified to be bit-identical to AT-DGNN. Every variant sees the
same fold split for a given seed, so the comparison is paired.

### Cross-subject evaluation (LOSO)

LOSO requires **one preprocessed file per subject** in `--data-folder`; produce them first:

```shell
# raw files must be named sample_1.dat ... sample_N.dat (MEEG) or s01.dat ... (DEAP)
python main.py --data-path /path/to/raw --dataset MEEG --sampling-rate=1000   --target-rate=200 --trial-duration=59 --input-shape '1,32,800' --subjects=32
```

```shell
python experiments/loso.py --data-folder data_eeg_MEEG_A   --variants "AT-DGNN" "AT-DGNN-BandAttn" "BandAttn (static)" "BandAttn (adaptive)"   --epochs 40 --val-subjects 2 --normalize per_subject   --out experiments/results/loso_MEEG.json
python experiments/summarize_loso.py --pattern 'experiments/results/loso_MEEG.json'
```

Each fold trains on all other subjects and tests on the held-out one, so the statistical unit is
the *subject* (n = number of subjects) and the comparison against the baseline is paired by
subject. Cost on an RTX 4060 Laptop for MEEG (31 training subjects, 8680 segments/epoch,
40 epochs, 32 folds): about 16 h per variant in fp32.
