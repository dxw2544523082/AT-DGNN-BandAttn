"""
Run only the data-preparation stage of the pipeline (no training).

Useful before a LOSO sweep, since main.py always runs the full cross-validation
after preparing the data.

    python experiments/prepare_dataset.py --data-path ./data_raw_DEAP/data_preprocessed_python \
        --dataset DEAP --subjects 32
"""
import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_argv = sys.argv
sys.argv = _argv[:1]
from config.config import set_config  # noqa: E402
from train.prepare_data import PrepareData  # noqa: E402
sys.argv = _argv


def main():
    a = sys.argv[1:]
    sys.argv = ['prepare'] + a
    args, _ = set_config()
    sys.argv = _argv
    subs = np.arange(args.subjects)
    print(f'>>> preparing {args.dataset}: {len(subs)} subjects from {args.data_path}')
    PrepareData(args).run(subs, split=True, expand=True)
    print('>>> done')


if __name__ == '__main__':
    main()
