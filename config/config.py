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
                        choices=['AT-DGNN', 'AT-DGNN-FixedLayout', 'AT-DGNN-AttGraph', 'AT-DGNN-TemporalAttn', 'AT-DGNN-BandAttn', 'LGGNet', 'EEGNet', 'DeepConvNet', 'ShallowConvNet',
                                 'EEG-TCNet', 'TSception', 'TCNet-Fusion', 'ATCNet', 'DGCNN'])
    parser.add_argument('--pool', type=int, default=16)
    parser.add_argument('--pool-step-rate', type=float, default=0.25)
    parser.add_argument('--T', type=int, default=64)
    parser.add_argument('--graph-type', type=str, default='fro', choices=['fro', 'gen', 'hem', 'BL'])
    parser.add_argument('--hidden', type=int, default=32)  # 隐藏层
    # band-attention variant: sample-adaptive weighting over canonical EEG bands
    parser.add_argument('--band-attn', type=str, default='both',
                        choices=['none', 'static', 'adaptive', 'both'],
                        help='static: learnable band prior only; adaptive: sample-dependent '
                             'modulation only; both: sum of the two (default)')
    parser.add_argument('--band-numtaps', type=int, default=129,
                        help='FIR length of the fixed band-split filters')
    parser.add_argument('--band-kind', type=str, default='fft', choices=['fft', 'fir'],
                        help='fft: brick-wall masks (zero band leakage, default); '
                             'fir: linear-phase FIR telescoping (wide transition bands)')
    parser.add_argument('--band-hidden', type=int, default=8,
                        help='hidden width of the shared score MLP (adaptive part)')

    # sliding-window tensor layout / implementation
    parser.add_argument('--sliding-layout', type=str, default='legacy', choices=['legacy', 'fixed'],
                        help="legacy: bit-compatible with the original (scrambled) flattening; "
                             "fixed: dimension 1 is the electrode axis again (see "
                             "SlidingWindowProcessor docstring)")
    parser.add_argument('--sliding-vectorize', dest='sliding_vectorized', action='store_true',
                        default=False,
                        help='opt-in only: batching all windows into one MHSA call was measured to be '
                             '~4.5x SLOWER than the original Python loop (1339 ms vs 301 ms per '
                             'fwd+bwd+step at batch 64), so the loop is the default. The batched '
                             'path is numerically equivalent (max abs diff 0.0) but slower.')
    parser.add_argument('--graph-hidden', type=int, default=-1,
                        help='width of the hidden layers inside the stacked graph convolution; '
                             '<=0 keeps the AT-DGNN default (input_shape[2]). Used for the '
                             'parameter-matched control experiments.')

    # AT-DGNN-AttGraph ablation switches (both default to the full model)
    parser.add_argument('--use-attn-graph', dest='use_attn_graph', action='store_true', default=True,
                        help='use the attention-generated dynamic adjacency A_att = softmax(H A H^T)')
    parser.add_argument('--no-attn-graph', dest='use_attn_graph', action='store_false',
                        help='ablation: fall back to the feature self-similarity adjacency of AT-DGNN')
    parser.add_argument('--use-global-attn', dest='use_global_attn', action='store_true', default=True,
                        help='use the global attention readout beta_i = softmax(omega_i)')
    parser.add_argument('--no-global-attn', dest='use_global_attn', action='store_false',
                        help='ablation: flatten the node features without global attention')
    # temporal-attention variant: multi-dimensional attention on the temporal
    # representation, followed by AT-DGNN's unmodified dynamic graph convolution
    parser.add_argument('--temporal-attn', type=str, default='factorized',
                        choices=['none', 'factorized', 'joint', 'sharp', 'sigmoid'],
                        help='factorized = one softmax per temporal sub-axis (window, position)')
    parser.add_argument('--temporal-dim', type=int, default=32,
                        help='projection dimension d of the temporal attention keys')
    parser.add_argument('--temporal-init-gain', type=float, default=1.0,
                        help="initial value of the learnable gain used by --temporal-attn sharp")
    parser.add_argument('--temporal-placement', type=str, default='pre', choices=['pre', 'post'],
                        help='pre: gate the (B,C,T) temporal representation; '
                             'post: gate the (B,N,T) node features right before the DGCN')
    parser.add_argument('--temporal-residual', dest='temporal_residual', action='store_true',
                        default=False, help='add an identity path to the temporal gate')
    parser.add_argument('--attn-self-loop', dest='attn_self_loop', action='store_true', default=False,
                        help='restore the explicit self-connection (A_att + I) that the plain '
                             'softmax adjacency tends to lose')

    # Reproduce the result using the saved model
    parser.add_argument('--reproduce', action='store_true', default=False)

    args = parser.parse_args()
    gpu = args.gpu
    # Convert the input shape from string to tuple of integers
    args.input_shape = tuple(map(int, args.input_shape.split(',')))

    return args, gpu
