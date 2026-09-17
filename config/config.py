import argparse


def set_config():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--dataset', type=str, default='MEEG', choices=['MEEG', 'DEAP'])
    parser.add_argument('--data-path', type=str, default='/home/data/eeg-data')
    parser.add_argument('--subjects', type=int, default=32)
    parser.add_argument('--num-class', type=int, default=2, choices=[2, 3, 4])
    parser.add_argument('--label-type', type=str, default='A', choices=['A', 'V', 'D', 'L'])
    parser.add_argument('--segment', type=int, default=4)  # segment length in seconds
    parser.add_argument('--overlap', type=float, default=0)
    parser.add_argument('--sampling-rate', type=int, default=1000)
    parser.add_argument('--target-rate', type=int, default=200)
    parser.add_argument('--trial-duration', type=int, default=59, help='trial duration in seconds')
    parser.add_argument('--input-shape', type=str, default="1,32,800") # 输入形状 (1, 32, 512)
    parser.add_argument('--data-format', type=str, default='eeg')
    parser.add_argument('--bandpass', type=tuple, default=(1, 50))
    parser.add_argument('--channels', type=int, default=32)

    # Training Process
    parser.add_argument('--fold', type=int, default=10)
    parser.add_argument('--random-seed', type=int, default=3407)
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--patient', type=int, default=40)  # 早停 最开始为20
    parser.add_argument('--patient-cmb', type=int, default=10)  # 原始值为8
    parser.add_argument('--max-epoch-cmb', type=int, default=40)  # 最大迭代次数 原始值为20
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-3)  # 学习率 原始值为1e-3
    parser.add_argument('--training-rate', type=float, default=0.8)
    parser.add_argument('--weight-decay', type=float, default=0.001)  # 权重衰减
    parser.add_argument('--step-size', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.5)  # 原始0.5
    parser.add_argument('--LS', type=bool, default=True, help="Label smoothing")  # 原始值为True
    parser.add_argument('--LS-rate', type=float, default=0.1)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--balance', type=bool, default=False)

    parser.add_argument('--save-path', default='./save/')
    parser.add_argument('--load-path', default='./save/max-acc.pth')
    parser.add_argument('--load-path-final', default='./save/final_model.pth')
    parser.add_argument('--save-model', type=bool, default=True)
    # Model Parameters
    parser.add_argument('--model', type=str, default='AT-DGNN',
                        choices=['AT-DGNN', 'LGGNet', 'EEGNet', 'DeepConvNet', 'ShallowConvNet', 'EEG-TCNet',
                                 'TSception', 'TCNet-Fusion', 'ATCNet', 'DGCNN',
                                 'AT-DGNN-BandAttn', 'AT-DGNN-ScaleAttn'])
    parser.add_argument('--pool', type=int, default=16)
    parser.add_argument('--pool-step-rate', type=float, default=0.25)
    parser.add_argument('--T', type=int, default=64)
    parser.add_argument('--graph-type', type=str, default='fro', choices=['fro', 'gen', 'hem', 'BL'])
    parser.add_argument('--hidden', type=int, default=32)  # 隐藏层

    # ---- multi-scale (frequency-scale) attention over the Tception branches ----
    parser.add_argument('--scale-attn', type=str, default='both',
                        choices=['none', 'static', 'adaptive', 'both', 'scalar'],
                        help='static: learnable branch prior only; adaptive: sample-dependent '
                             'modulation; both: their sum (default); scalar: one learned weight '
                             'per branch, shared by all samples')
    parser.add_argument('--scale-hidden', type=int, default=8,
                        help='hidden width of the shared branch-score MLP')

    # ---- band attention (this work) ----
    parser.add_argument('--band-attn', type=str, default='both',
                        choices=['none', 'static', 'adaptive', 'both'],
                        help='static: learnable band prior only; adaptive: sample-dependent '
                             'modulation only; both: their sum (default); none: baseline')
    parser.add_argument('--band-kind', type=str, default='fft', choices=['fft', 'fir'],
                        help='fft: brick-wall masks (no inter-band leakage, default); '
                             'fir: linear-phase FIR telescoping')
    parser.add_argument('--band-numtaps', type=int, default=129,
                        help='FIR length when --band-kind fir')
    parser.add_argument('--band-fuse', type=str, default='residual',
                        choices=['residual', 'replace'],
                        help='residual: x + sum(beta_k-1)*X_k (identity when beta==1, '
                             'default); replace: sum(beta_k*X_k), only approximately x')
    parser.add_argument('--band-hidden', type=int, default=8,
                        help='hidden width of the shared band-score MLP')

    # Reproduce the result using the saved model
    parser.add_argument('--reproduce', action='store_true', default=False)

    args = parser.parse_args()
    gpu = args.gpu
    # Convert the input shape from string to tuple of integers
    args.input_shape = tuple(map(int, args.input_shape.split(',')))

    return args, gpu
