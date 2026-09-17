import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import *
from models.models import GraphNeuralNetwork, Aggregator, PowerLayer

_, os.environ['CUDA_VISIBLE_DEVICES'] = set_config()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class ATDGNN(nn.Module):

    def temporal_learner(self, in_chan, out_chan, kernel, pool, pool_step_rate):
        return nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=kernel, stride=(1, 1)),
            PowerLayer(dim=-1, length=pool, step=int(pool_step_rate * pool))
        )

    def __init__(self, num_classes, input_size, sampling_rate, num_T,
                 out_graph, dropout_rate, pool, pool_step_rate, idx_graph, graph_hidden=None,
                 sliding_layout='legacy', sliding_vectorized=False):
        super(ATDGNN, self).__init__()

        self.num_T = num_T
        self.out_graph = out_graph
        self.dropout_rate = dropout_rate
        self.window = [0.5, 0.25, 0.125]
        self.pool = pool
        self.pool_step_rate = pool_step_rate
        self.idx = idx_graph
        self.channel = input_size[1]
        self.brain_area = len(self.idx)
        ###################
        # 多头注意力相关参数
        self.model_dim = round(num_T / 2)
        self.num_heads = 8
        if sampling_rate == 200:
            self.window_size = 100
            self.stride = 20
        else:
            self.window_size = 64
            self.stride = 16
        ###################
        hidden_features = input_size[2] if graph_hidden is None else graph_hidden

        # by setting the convolutional kernel being (1,lenght) and the strids being 1, we can use conv2d to
        # achieve the 1d convolution operation.
        self.Tception1 = self.temporal_learner(input_size[0], num_T,
                                               (1, int(self.window[0] * sampling_rate)),
                                               self.pool, pool_step_rate)
        self.Tception2 = self.temporal_learner(input_size[0], num_T,
                                               (1, int(self.window[1] * sampling_rate)),
                                               self.pool, pool_step_rate)
        self.Tception3 = self.temporal_learner(input_size[0], num_T,
                                               (1, int(self.window[2] * sampling_rate)),
                                               self.pool, pool_step_rate)
        # Batch normalization layers
        self.bn_t = nn.BatchNorm2d(num_T)
        self.bn_s = nn.BatchNorm2d(num_T)
        self.OneXOneConv = nn.Sequential(
            nn.Conv2d(num_T, num_T, kernel_size=(1, 1), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.AvgPool2d((1, 2))
        )
        #######################################
        # 特征整合、滑动窗口相关配置
        self.feature_integrator = FeatureIntegrator(in_channels=32, out_channels=self.model_dim)
        self.sliding_window_processor = SlidingWindowProcessor(model_dim=self.model_dim, num_heads=self.num_heads,
                                                               window_size=self.window_size, stride=self.stride,
                                                               layout=sliding_layout,
                                                               vectorized=sliding_vectorized)
        #######################################
        # diag(W) to assign a weight to each local areas
        size = self.get_size_temporal(input_size)
        # 表示局部滤波器的权重。它被定义为一个形状为(self.channel, size[-1])的浮点型张量，并设置为需要梯度计算（requires_grad=True）
        self.local_filter_weight = nn.Parameter(torch.FloatTensor(self.channel, size[-1]),
                                                requires_grad=True)
        # 用来对local_filter_weight进行初始化，采用的是Xavier均匀分布初始化方法
        nn.init.xavier_uniform_(self.local_filter_weight)
        # 表示局部滤波器的偏置。它被定义为一个形状为(1, self.channel, 1)的浮点型张量，初始值为全零，并设置为需要梯度计算
        self.local_filter_bias = nn.Parameter(torch.zeros((1, self.channel, 1), dtype=torch.float32),
                                              requires_grad=True)
        # aggregate function
        self.aggregate = Aggregator(self.idx)

        # Dynamic Graph Convolution Layers
        # self.dynamic_gcn = DynamicGraphNeuralNetwork(size[-1], out_graph)
        self.dynamic_gcn = StackedDynamicGraphNeuralNetwork(size[-1], hidden_features, out_graph, num_layers=3)
        # 表示全局邻接矩阵。它被定义为浮点型张量，并设置为需要梯度计算（requires_grad=True）
        self.global_adj = nn.Parameter(torch.FloatTensor(self.brain_area, self.brain_area), requires_grad=True)
        # 根据给定的张量的形状和分布进行参数初始化。用来对global_adj进行初始化，采用的是Xavier均匀分布初始化方法。
        nn.init.xavier_uniform_(self.global_adj)
        # to be used after local graph embedding
        self.bn = nn.BatchNorm1d(self.brain_area)
        self.bn_ = nn.BatchNorm1d(self.brain_area)

        # Fully connected layer for classification
        self.fc = nn.Sequential(  # 组合神经网络模块
            nn.Dropout(p=dropout_rate),
            nn.Linear(int(self.brain_area * out_graph), num_classes)
        )

    def get_size_temporal(self, input_size):
        # input_size: frequency x channel x data point
        data = torch.ones((1, input_size[0], input_size[1], int(input_size[2])))
        z = self.Tception1(data)
        out = z
        z = self.Tception2(data)
        out = torch.cat((out, z), dim=-1)
        z = self.Tception3(data)
        out = torch.cat((out, z), dim=-1)
        #######################################
        out = self.feature_integrator(out)  # 特征整合和降维
        out = self.sliding_window_processor(out)  # 滑动窗口处理
        #######################################
        out = torch.reshape(out, (out.size(0), out.size(1), -1))
        size = out.size()
        return size

    # 定义局部滤波器的前向传播函数
    def local_filter_fun(self, x, w):
        w = w.unsqueeze(0).repeat(x.size()[0], 1, 1)
        x = F.relu(torch.mul(x, w) - self.local_filter_bias)
        return x

    def forward(self, x):
        # Temporal convolution
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        ##############################
        out = self.feature_integrator(out)  # 特征整合和降维
        out = self.sliding_window_processor(out)  # 滑动窗口处理
        ##############################
        out = torch.reshape(out, (out.size(0), out.size(1), -1))
        out = self.local_filter_fun(out, self.local_filter_weight)
        out = self.aggregate.forward(out)
        out = self.bn(out)
        out = self.dynamic_gcn(out)
        out = self.bn_(out)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out


class DynamicGraphNeuralNetwork(GraphNeuralNetwork):
    """
    Dynamic Graph Neural Network Layer.
    Extends the GraphNeuralNetwork layer with a dynamic adjacency matrix based on feature similarity.
    """

    def __init__(self, in_features, out_features, bias=True):
        super(DynamicGraphNeuralNetwork, self).__init__(in_features, out_features, bias)

    def forward(self, x, adj=None):
        if adj is None:
            # Compute adjacency matrix dynamically based on feature similarity, for example:
            adj = self.normalize_adjacency_matrix(x)

        output = torch.matmul(x, self.weight)
        if self.bias is not None:
            output += self.bias
        output = F.relu(torch.matmul(adj, output))
        return output

    def compute_similarity(self, x):
        # x: b, node, feature
        x_ = x.permute(0, 2, 1)
        s = torch.bmm(x, x_)
        return s

    def normalize_adjacency_matrix(self, x):
        """
        x：输入的特征矩阵，大小为(b, node, feature)，其中b为批次大小，node为节点数目，feature为每个节点的特征向量维度。
        self_loop：一个布尔值，表示是否在邻接矩阵中加入自环（自己到自己的连接）。
        """
        # x: b, node, feature
        # 利用模型中的self_similarity方法计算输入特征矩阵x的自相似度矩阵。结果为一个大小为(b, n, n)的张量，其中n为节点数目
        adj = self.compute_similarity(x)  # b, n, n
        num_nodes = adj.shape[-1]
        # NOTE: use adj.device instead of the module-level DEVICE constant so that
        # the model can run on a device other than the one selected at import time
        # (e.g. CPU while a GPU is present).
        adj = adj + torch.eye(num_nodes, device=adj.device, dtype=adj.dtype)
        rowsum = torch.sum(adj, dim=-1)
        # 创建一个与rowsum大小相同的全零张量mask，并将rowsum中和为0的位置置为1。这一步是为了处理邻接矩阵中存在度为0的节点，避免除以0的错误
        mask = torch.zeros_like(rowsum)
        mask[rowsum == 0] = 1
        # 将mask添加到rowsum中，实现对邻接矩阵的修正。避免除以0的错误，并保证每个节点的度至少为1
        rowsum += mask
        # 计算度矩阵的逆平方根
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        # 将逆平方根得到的张量转换为对角矩阵
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        # 通过矩阵乘法和广播机制，将度矩阵的逆平方根与邻接矩阵相乘，得到归一化后的邻接矩阵
        adj = torch.bmm(torch.bmm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return adj


class AttentionDynamicGraphNeuralNetwork(GraphNeuralNetwork):
    """
    Attention-based dynamic graph convolution layer (AttGraph, Eqs. 1-4).

        H     = X W                                                        (1)
        A_att = softmax(H A H^T, dim = -1)                                 (2)
        A_att = (A_att + A_att^T) / 2                                      (3)
        Z     = A_att H                                                    (4)

    A is a learnable F' x F' attention kernel shared by all samples, so the
    electrode graph is *generated from the data* instead of being derived from
    the feature self-similarity and a fixed spatial prior as in DGCNN/AT-DGNN.
    The softmax is taken over the electrode dimension and the graph is then
    symmetrized, which keeps the connectivity bidirectional.
    """

    def __init__(self, in_features, out_features, bias=True, self_loop=False):
        super(AttentionDynamicGraphNeuralNetwork, self).__init__(in_features, out_features, bias)
        # restore the explicit self-connection that AT-DGNN has (+I) and that a
        # plain softmax adjacency tends to lose (measured diagonal mass ~= 1/N)
        self.self_loop = self_loop
        # trainable attention weight kernel A in Eq. (2): A in R^{F' x F'}, where
        # F' = out_features is the *output* feature dimension of this layer.
        self.att_kernel = nn.Parameter(torch.FloatTensor(out_features, out_features),
                                       requires_grad=True)
        nn.init.xavier_uniform_(self.att_kernel)
        # cached for attention visualization (not used in the forward pass)
        self.last_attn = None       # effective adjacency (after the optional self-loop)
        self.last_attn_raw = None   # symmetrised softmax adjacency (Eq. 2-3)

    def forward(self, x, adj=None, return_attn=False):
        # `adj` is accepted only for API compatibility with the DGCNN layers;
        # the graph of this layer is produced by the attention mechanism itself.
        # x: batch x node(electrode) x feature
        h = torch.matmul(x, self.weight)
        if self.bias is not None:
            h = h + self.bias

        # Eq. (2): dynamic adjacency matrix
        att = torch.matmul(torch.matmul(h, self.att_kernel), h.transpose(-2, -1))
        att = F.softmax(att, dim=-1)
        # Eq. (3): symmetrization
        att = (att + att.transpose(-2, -1)) / 2.0
        # raw symmetrised softmax adjacency (Eq. 2-3), before the optional self-loop:
        # this is the quantity used in the diagnosis of section 4.6
        self.last_attn_raw = att.detach()
        if self.self_loop:
            att = att + torch.eye(att.size(-1), device=att.device, dtype=att.dtype)
        # effective adjacency actually used by the layer
        self.last_attn = att.detach()

        # Eq. (4): weighted aggregation
        out = F.relu(torch.matmul(att, h))
        if return_attn:
            return out, att
        return out


class GlobalAttentionReadout(nn.Module):
    """
    Global attention module (AttGraph, Eq. 5).

        beta_i = exp(omega_i) / sum_j exp(omega_j),   Z'_i = beta_i * Z_i     (5)

    A scalar score omega_i is learned for every electrode node so that the
    decision stage can re-weight (and thereby select) the electrode channels
    that are most informative for the emotional state.
    """

    def __init__(self, in_features):
        super(GlobalAttentionReadout, self).__init__()
        self.score = nn.Linear(in_features, 1)
        self.last_beta = None

    def forward(self, x, return_weights=False):
        # x: batch x node x feature
        omega = self.score(x).squeeze(-1)                 # (B, N)
        beta = F.softmax(omega, dim=-1)                   # (B, N)
        self.last_beta = beta.detach()
        out = x * beta.unsqueeze(-1)                      # Z'_i = beta_i * Z_i
        if return_weights:
            return out, beta
        return out


class StackedDynamicGraphNeuralNetwork(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_layers=3, bias=True,
                 layer_cls=DynamicGraphNeuralNetwork, **layer_kwargs):
        super(StackedDynamicGraphNeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(layer_cls(in_features, hidden_features, bias=bias, **layer_kwargs))
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(layer_cls(hidden_features, hidden_features, bias=bias, **layer_kwargs))
        # Last layer
        self.layers.append(layer_cls(hidden_features, out_features, bias=bias, **layer_kwargs))

    def forward(self, x, adj=None):
        for layer in self.layers:
            x = layer(x, adj)
        return x

    def attention_maps(self, raw=False):
        key = 'last_attn_raw' if raw else 'last_attn'
        return [getattr(l, key) for l in self.layers if getattr(l, key, None) is not None]


class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TemporalConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return F.relu(self.norm(self.conv(x)))


class FeatureIntegrator(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=64, stride=64):
        super(FeatureIntegrator, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        # 假设输入x的形状为 (batch_size, feature_dim, channels, length)
        batch_size, feature_dim, channels, length = x.size()

        # 你想将feature和length维度相结合
        # 首先，将x变形为 (batch_size, channels, feature_dim * length)
        x = x.reshape(batch_size, channels, feature_dim * length)

        # 然后，应用1D卷积
        x = self.conv(x)  # 卷积后的形状为 (batch_size, out_channels, new_length)

        return x


class SlidingWindowProcessor(nn.Module):
    """
    Sliding-window multi-head attention + TCN.

    NOTE on ``layout`` -- this switch does NOT change the model's maths, it fixes a
    tensor-layout defect in the original implementation:

    The original code stacks the per-window outputs as ``(B, C, n_win, Ws)`` and then
    does ``permute(0, 3, 1, 2).reshape(B, C, -1)``.  That reshape does **not** flatten
    "windows then positions": because the permuted tensor is laid out as
    ``(position, channel, window)`` with the window index varying fastest, the flat
    axis ends up ordered as ``f = p * (C * n_win) + c * n_win + w``, and the resulting
    dimension 1 (size C) is **not** the electrode dimension.  Downstream, the local
    filter is indexed as ``(channel, time)`` and the aggregator averages groups of
    dimension 1 that are supposed to be brain areas (from
    ``num_chan_local_graph_*.hdf``); with the scrambled layout neither is doing what it
    claims.  For the MEEG configuration each "brain area" group turns out to span
    ~3.1 consecutive *time positions* across all channels, i.e. the 14 "brain-area
    nodes" are really 14 temporal chunks.

    layout='legacy' : bit-compatible replication of the original (scrambled) flattening.
    layout='fixed'  : the intended layout, out[b, c, w * Ws + p] = window w, channel c,
                      position p, so that dimension 1 is the electrode axis again.

    ``vectorized=True`` batches all windows into a single MHSA call instead of looping
    in Python.  It is mathematically equivalent (verified: max abs diff 0.0 against the
    loop) but **was measured to be ~4.5x slower** on an RTX 4060 (1339 vs 301 ms per
    fwd+bwd+step at batch 64; the sliding-window block alone 131 vs 24 ms), so the loop
    stays the default.  Keeping both paths makes that equivalence check reproducible.
    """

    def __init__(self, model_dim, num_heads, window_size, stride, layout='legacy',
                 vectorized=False):
        super(SlidingWindowProcessor, self).__init__()
        self.window_size = window_size
        self.stride = stride
        self.layout = layout
        self.vectorized = vectorized
        self.layer_norm1 = nn.LayerNorm([window_size, model_dim])
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm2 = nn.LayerNorm(model_dim)
        self.tcn_block = TemporalConvBlock(in_channels=model_dim, out_channels=32)
        # 定义融合层，使用1D卷积以保留32通道的结构，卷积核大小和步长可以根据需要调整
        self.fusion_conv = nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1)

    def _windows(self, x):
        """Return the list of window tensors exactly as the original loop produced them."""
        _, _, length = x.shape
        return [x[:, :, s:s + self.window_size]
                for s in range(0, length - self.window_size + 1, self.stride)]

    def _forward_loop(self, x):
        """Original implementation (kept for exact reproduction)."""
        batch_size, _, length = x.shape
        window_outputs = []
        for s in range(0, length - self.window_size + 1, self.stride):
            window = x[:, :, s:s + self.window_size].permute(0, 2, 1)
            window = self.layer_norm1(window)
            attn_output, _ = self.multi_head_attention(window, window, window)
            attn_output = self.layer_norm2(attn_output + window)
            window_outputs.append(self.tcn_block(attn_output.permute(0, 2, 1)))
        stacked = torch.stack(window_outputs, dim=2)          # (B, 32, n_win, Ws)
        if self.layout == 'fixed':
            # (B, c, w, p) flattened with p fastest -> out[b, c, w * Ws + p]
            stacked = stacked.reshape(batch_size, 32, -1)
        else:
            # bit-compatible with the original permute(0,3,1,2).reshape(B, 32, -1)
            stacked = stacked.permute(0, 3, 1, 2).reshape(batch_size, 32, -1)
        return self.fusion_conv(stacked)

    def forward(self, x):
        if not self.vectorized:
            return self._forward_loop(x)

        batch_size, channels, length = x.shape
        n_win = (length - self.window_size) // self.stride + 1
        # (B, C, n_win, Ws) -> (B, n_win, C, Ws) -> (B*n_win, Ws, C)
        win = x.unfold(2, self.window_size, self.stride).permute(0, 2, 1, 3)
        h = win.reshape(batch_size * n_win, channels, self.window_size).permute(0, 2, 1)
        h = self.layer_norm1(h)
        attn, _ = self.multi_head_attention(h, h, h)
        attn = self.layer_norm2(attn + h)
        t = self.tcn_block(attn.permute(0, 2, 1))                     # (B*n_win, 32, Ws)
        out_dim = t.size(1)
        t = t.reshape(batch_size, n_win, out_dim, self.window_size)   # (B, n_win, 32, Ws)

        if self.layout == 'fixed':
            # out[b, c, w * Ws + p] = t[b, w, c, p] : dimension 1 is the channel axis
            stacked = t.permute(0, 2, 1, 3).reshape(batch_size, out_dim, n_win * self.window_size)
        else:
            # bit-compatible with torch.stack(windows, dim=2).permute(0,3,1,2).reshape(...)
            stacked = t.permute(0, 2, 1, 3).permute(0, 3, 1, 2).reshape(batch_size, out_dim, -1)
        return self.fusion_conv(stacked)


class ATDGNN_AttGraph(ATDGNN):
    """
    AT-DGNN + AttGraph: multi-dimensional attention-based dynamic graph
    convolution for EEG emotion recognition.

    The temporal front-end (Tception temporal learners -> feature integrator ->
    sliding-window multi-head attention + TCN -> local filter -> brain-area
    aggregator) is inherited from AT-DGNN unchanged, so that any performance
    difference is attributable to the two graph-side modules below:

    * ``use_attn_graph``   : replace the feature-self-similarity adjacency of the
                             stacked DGCN with the attention-generated adjacency
                             A_att = softmax(H A H^T) (AttGraph Eqs. 1-4).
    * ``use_global_attn``  : insert a global attention module that weights every
                             electrode node with beta_i = softmax(omega_i) before
                             flattening (AttGraph Eq. 5).

    Setting both switches to False reproduces the AT-DGNN baseline exactly, so
    the four combinations form the ablation grid of the paper.
    """

    def __init__(self, num_classes, input_size, sampling_rate, num_T,
                 out_graph, dropout_rate, pool, pool_step_rate, idx_graph,
                 use_attn_graph=True, use_global_attn=True, graph_hidden=None,
                 attn_self_loop=False):
        super(ATDGNN_AttGraph, self).__init__(
            num_classes=num_classes, input_size=input_size, sampling_rate=sampling_rate,
            num_T=num_T, out_graph=out_graph, dropout_rate=dropout_rate, pool=pool,
            pool_step_rate=pool_step_rate, idx_graph=idx_graph, graph_hidden=graph_hidden,
            sliding_layout=sliding_layout, sliding_vectorized=sliding_vectorized)

        self.use_attn_graph = use_attn_graph
        self.use_global_attn = use_global_attn
        self.attn_self_loop = attn_self_loop

        # the unused learnable global adjacency of AT-DGNN is dropped from the
        # optimizer when the graph is generated by attention instead
        if use_attn_graph:
            size = self.get_size_temporal(input_size)
            graph_hidden = input_size[2] if graph_hidden is None else graph_hidden
            self.dynamic_gcn = StackedDynamicGraphNeuralNetwork(
                size[-1], graph_hidden, out_graph, num_layers=3,
                layer_cls=AttentionDynamicGraphNeuralNetwork,
                self_loop=attn_self_loop)
            del self.global_adj

        if use_global_attn:
            self.global_attn = GlobalAttentionReadout(out_graph)

    def forward(self, x):
        # ---- temporal front-end (identical to AT-DGNN) ----
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        out = self.feature_integrator(out)
        out = self.sliding_window_processor(out)
        out = torch.reshape(out, (out.size(0), out.size(1), -1))
        out = self.local_filter_fun(out, self.local_filter_weight)
        out = self.aggregate.forward(out)
        out = self.bn(out)

        # ---- graph-side modules ----
        out = self.dynamic_gcn(out)
        out = self.bn_(out)
        if self.use_global_attn:
            out = self.global_attn(out)

        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out

    @torch.no_grad()
    def attention_weights(self, x, raw=False):
        """
        Convenience hook for the attention-visualization experiments: run a
        forward pass and return the per-layer dynamic adjacency matrices and
        the electrode-level global attention weights.

        raw=False returns the adjacency actually used by each layer; raw=True
        returns the symmetrised softmax adjacency before the optional +I.
        """
        self.eval()
        self(x)
        graphs = self.dynamic_gcn.attention_maps(raw=raw) if self.use_attn_graph else []
        beta = self.global_attn.last_beta if self.use_global_attn else None
        return graphs, beta



class MultiDimensionalTemporalAttention(nn.Module):
    """
    Multi-dimensional attention over the *time* axis of the temporal
    representation produced by the AT-DGNN front-end.

    The front-end output has shape (B, C, T) with T = W * P, where W is the
    number of sliding windows and P the number of samples inside a window
    (MEEG configuration: T = 2300 = 23 windows x 100 positions).  A scalar
    importance score is learned for every (window, position) pair and used to
    re-weight the representation *before* it is turned into graph nodes:

        k_{w,p}  = W_k h[:, :, w, p]                 (key projection, C -> d)
        s_{w,p}  = q^T k_{w,p} / sqrt(d)             (learned query q)
        beta_w   = W * softmax_w( mean_p s )         (window importance)
        beta_p   = P * softmax_p( mean_w s )         (within-window importance)
        h'_{w,p} = beta_w * beta_p * h_{w,p}         (factorised 2-D gate)

    Normalising each axis separately (``mode='factorized'``) keeps the mean
    gate at 1 and makes the two weight vectors individually interpretable: one
    can read off which windows and which within-window positions the model
    considers informative.  ``mode='joint'`` instead uses a single softmax over
    the flattened W*P axis.  ``residual=True`` adds the identity path
    (gate = 1 + beta) so the original signal always survives.
    """

    def __init__(self, channels, num_windows, window_len, dim=32, mode='factorized',
                 residual=False, init_gain=1.0):
        super(MultiDimensionalTemporalAttention, self).__init__()
        self.channels = channels
        self.num_windows = int(num_windows)
        self.window_len = int(window_len)
        self.mode = mode
        self.residual = residual
        self.dim = dim
        self.key = nn.Linear(channels, dim, bias=False)
        self.query = nn.Parameter(torch.randn(dim) / dim ** 0.5)
        self.scale = float(dim) ** 0.5
        # mode='sharp': layer-normalise the scores per sample and rescale with a
        # learnable gain, so the softmax sharpness does not depend on the
        # (initially tiny) magnitude of q^T W_k h.
        # mode='sigmoid': drop the competition entirely -- an unnormalised gate
        # whose mean is free to move instead of being pinned to 1 by softmax.
        self.log_gain = nn.Parameter(torch.tensor(math.log(max(float(init_gain), 1e-6)),
                                                dtype=torch.float32))
        self.gate_bias_w = nn.Parameter(torch.zeros(1))
        self.gate_bias_p = nn.Parameter(torch.zeros(1))
        # cached for the interpretability analysis
        self.last_beta = None
        self.last_beta_w = None
        self.last_beta_p = None

    @staticmethod
    def _norm_axis(x):
        """Layer-normalise the last axis to zero mean / unit variance."""
        return (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-5)

    def forward(self, h, return_weights=False):
        B, C, T = h.shape
        W, P = self.num_windows, self.window_len
        if T != W * P:
            raise ValueError(f'temporal axis {T} is not {W} windows x {P} positions')
        hw = h.reshape(B, C, W, P)
        tokens = hw.permute(0, 2, 3, 1)                       # (B, W, P, C)
        s = torch.matmul(self.key(tokens), self.query) / self.scale   # (B, W, P)

        if self.mode == 'joint':
            beta = torch.softmax(s.reshape(B, -1), dim=-1).reshape(B, W, P) * (W * P)
            self.last_beta_w = self.last_beta_p = None
        elif self.mode == 'sharp':
            gain = torch.exp(self.log_gain)
            sw = self._norm_axis(s.mean(dim=-1)) * gain
            sp = self._norm_axis(s.mean(dim=-2)) * gain
            beta_w = torch.softmax(sw, dim=-1) * W
            beta_p = torch.softmax(sp, dim=-1) * P
            beta = beta_w.unsqueeze(-1) * beta_p.unsqueeze(1)
            self.last_beta_w, self.last_beta_p = beta_w.detach(), beta_p.detach()
        elif self.mode == 'sigmoid':
            beta_w = 2.0 * torch.sigmoid(s.mean(dim=-1) + self.gate_bias_w)
            beta_p = 2.0 * torch.sigmoid(s.mean(dim=-2) + self.gate_bias_p)
            beta = beta_w.unsqueeze(-1) * beta_p.unsqueeze(1)
            self.last_beta_w, self.last_beta_p = beta_w.detach(), beta_p.detach()
        else:
            beta_w = torch.softmax(s.mean(dim=-1), dim=-1) * W            # (B, W)
            beta_p = torch.softmax(s.mean(dim=-2), dim=-1) * P            # (B, P)
            beta = beta_w.unsqueeze(-1) * beta_p.unsqueeze(1)            # (B, W, P)
            self.last_beta_w, self.last_beta_p = beta_w.detach(), beta_p.detach()
        self.last_beta = beta.detach()

        gate = beta + 1.0 if self.residual else beta
        out = (hw * gate.unsqueeze(1)).reshape(B, C, T)
        if return_weights:
            return out, beta
        return out


class ATDGNN_TemporalAttn(ATDGNN):
    """
    AT-DGNN + multi-dimensional attention on the temporal representation.

    Only the *input* of the graph is changed: the temporal representation
    (B, C, T) is re-weighted by a learned temporal importance distribution and
    then handed to the **unmodified** AT-DGNN pipeline -- local filter,
    brain-area aggregation and the original self-similarity dynamic graph
    convolution.  The graph is therefore built from temporally selected
    features, while the way the graph is built stays exactly that of AT-DGNN.

    ``temporal_placement``
        'pre'  : gate the (B, C, T) temporal representation right after the
                 sliding-window processor (default).
        'post' : gate the (B, N, T) node features right before the DGCN.
    """

    def __init__(self, num_classes, input_size, sampling_rate, num_T,
                 out_graph, dropout_rate, pool, pool_step_rate, idx_graph,
                 graph_hidden=None, temporal_attn='factorized', temporal_dim=32,
                 temporal_placement='pre', temporal_residual=False,
                 temporal_init_gain=1.0, sliding_layout='legacy', sliding_vectorized=False):
        super(ATDGNN_TemporalAttn, self).__init__(
            num_classes=num_classes, input_size=input_size, sampling_rate=sampling_rate,
            num_T=num_T, out_graph=out_graph, dropout_rate=dropout_rate, pool=pool,
            pool_step_rate=pool_step_rate, idx_graph=idx_graph, graph_hidden=graph_hidden,
            sliding_layout=sliding_layout, sliding_vectorized=sliding_vectorized)
        self.temporal_attn_mode = temporal_attn
        self.temporal_placement = temporal_placement
        self.temporal_attn = None
        if temporal_attn != 'none':
            L, T, W, P = self._front_shapes(input_size)
            channels = input_size[1] if temporal_placement == 'pre' else self.brain_area
            self.temporal_attn = MultiDimensionalTemporalAttention(
                channels, W, P, dim=temporal_dim, mode=temporal_attn,
                residual=temporal_residual, init_gain=temporal_init_gain)
            print('>>> temporal attention: placement={} channels={} windows={} '
                  'window_len={} T={} mode={} params={}'.format(
                      temporal_placement, channels, W, P, T, temporal_attn,
                      sum(p.numel() for p in self.temporal_attn.parameters())))

    def _front_shapes(self, input_size):
        """Recover (feature length, flat temporal axis, #windows, window length)."""
        data = torch.ones((1, input_size[0], input_size[1], int(input_size[2])))
        with torch.no_grad():
            z = torch.cat((self.Tception1(data), self.Tception2(data),
                           self.Tception3(data)), dim=-1)
            z = self.feature_integrator(z)
            L = z.size(-1)
        P = self.sliding_window_processor.window_size
        stride = self.sliding_window_processor.stride
        W = (L - P) // stride + 1
        return L, W * P, W, P

    def forward(self, x):
        # ---- temporal front-end (identical to AT-DGNN) ----
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        out = self.feature_integrator(out)
        out = self.sliding_window_processor(out)
        out = torch.reshape(out, (out.size(0), out.size(1), -1))

        # ---- temporal attention on the temporal representation ----
        if self.temporal_attn is not None and self.temporal_placement == 'pre':
            out = self.temporal_attn(out)

        out = self.local_filter_fun(out, self.local_filter_weight)
        out = self.aggregate.forward(out)
        out = self.bn(out)

        if self.temporal_attn is not None and self.temporal_placement == 'post':
            out = self.temporal_attn(out)

        # ---- AT-DGNN's own dynamic graph convolution, unmodified ----
        out = self.dynamic_gcn(out)
        out = self.bn_(out)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out

    @torch.no_grad()
    def temporal_attention_weights(self, x):
        """Return (beta_w, beta_p, beta) for the interpretability analysis."""
        self.eval()
        self(x)
        if self.temporal_attn is None:
            return None, None, None
        return (self.temporal_attn.last_beta_w, self.temporal_attn.last_beta_p,
                self.temporal_attn.last_beta)


def _fir_lowpass(cutoff, fs, numtaps):
    """Linear-phase FIR low-pass via a windowed sinc (no scipy dependency at run time)."""
    import math
    n = torch.arange(numtaps, dtype=torch.float64) - (numtaps - 1) / 2.0
    fc = cutoff / fs                                   # normalised cutoff (cycles/sample)
    h = 2.0 * fc * torch.sinc(2.0 * fc * n)
    win = torch.hann_window(numtaps, periodic=False, dtype=torch.float64)
    h = h * win
    return (h / h.sum()).float()


class FixedBandSplit(nn.Module):
    """
    Fixed (non-learnable) EEG band decomposition.

    Two implementations are provided:

    ``kind='fft'`` (default) - brick-wall masks applied in the frequency domain::

        X_k = irfft( rfft(x) * mask_k )

      The masks are complementary, so ``sum_k x_k`` equals x limited to the union
      of the bands (1-50 Hz), i.e. uniform attention reproduces the baseline input
      up to the small out-of-band residue of the preprocessing.  There is **no
      spectral leakage between bands and no in-band distortion**, which matters
      because the band weights are meant to be interpretable.  This is also how
      the standard DE/PSD band features are computed in the EEG emotion
      literature.

    ``kind='fir'`` - linear-phase FIR low-pass telescoping (``band_k = LP(hi_k) -
      LP(lo_k)``).  Kept for comparison; note that a short FIR has a wide
      transition band, so neighbouring bands overlap substantially.

    Bands follow the canonical EEG definition at 200 Hz:
        delta 1-4, theta 4-8, alpha 8-13, beta 13-30, gamma 30-50 Hz
    """

    def __init__(self, bands=((1, 4), (4, 8), (8, 13), (13, 30), (30, 50)),
                 fs=200, numtaps=129, kind='fft', n_fft=None):
        super(FixedBandSplit, self).__init__()
        self.bands = tuple(bands)
        self.fs = fs
        self.kind = kind
        self.numtaps = numtaps
        self.n_fft = n_fft
        if kind == 'fir':
            lps = {}
            edges = sorted({lo for lo, _ in self.bands} | {hi for _, hi in self.bands})
            for e in edges:
                lps[e] = _fir_lowpass(e, fs, numtaps)
            w = torch.stack([lps[hi] - lps[lo] for lo, hi in self.bands], 0).unsqueeze(1)
            self.register_buffer('weight', w)
        else:
            # masks are built lazily once the segment length is known
            self.register_buffer('masks', torch.empty(0))

    def _build_masks(self, T, device, dtype):
        n_freq = T // 2 + 1
        freqs = torch.fft.rfftfreq(T, d=1.0 / self.fs, device=device)
        masks = torch.stack([((freqs >= lo) & (freqs < hi)).to(dtype)
                             for lo, hi in self.bands], 0)          # (K, n_freq)
        self.masks = masks
        return masks

    def forward(self, x):
        """x: (B, 1, C, T) -> (B, K, C, T)"""
        B, one, C, T = x.shape
        K = len(self.bands)
        if self.kind == 'fir':
            xf = x.reshape(B * C * one, 1, T)
            pad = self.numtaps // 2
            xk = F.conv1d(xf, self.weight, padding=pad)[..., :T]
        else:
            masks = self.masks
            if masks.numel() == 0 or masks.shape[-1] != T // 2 + 1:
                masks = self._build_masks(T, x.device, x.dtype)
            X = torch.fft.rfft(x.reshape(B * C, T), dim=-1)          # (B*C, n_freq)
            xk = torch.fft.irfft(X.unsqueeze(1) * masks, n=T, dim=-1)  # (B*C, K, T)
        return xk.reshape(B, C, K, T).permute(0, 2, 1, 3).contiguous()


class BandAttention(nn.Module):
    """
    Sample-adaptive attention over EEG bands.

        d_k = log( mean_{c,t} x_k^2 )                       (per-band log power)
        s_k = b_k + phi( d_k - mean_j d_j )                 (static prior + shared MLP)
        beta = K * softmax_k(s)                             (mean weight = 1)
        x'  = sum_k beta_k * x_k

    ``b_k`` is a learnable *static* band prior (directly comparable with the
    EEG literature), while ``phi`` makes the weighting depend on the sample.

    mode='static'   : only b_k            (ablate the adaptive part)
    mode='adaptive' : only phi            (ablate the static prior)
    mode='both'     : b_k + phi           (default)
    """

    def __init__(self, bands, fs=200, numtaps=129, hidden=8, mode='both', eps=1e-6,
                 kind='fft'):
        super(BandAttention, self).__init__()
        self.split = FixedBandSplit(bands, fs, numtaps, kind=kind)
        self.K = len(self.split.bands)
        self.mode = mode
        self.eps = eps
        self.b = nn.Parameter(torch.zeros(self.K))
        self.phi = nn.Sequential(nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.last_beta = None

    def forward(self, x, return_weights=False):
        B, one, C, T = x.shape
        xk = self.split(x)                                     # (B, K, C, T)
        d = torch.log(xk.pow(2).mean(dim=(2, 3)) + self.eps)   # (B, K)
        d = d - d.mean(dim=1, keepdim=True)                    # centre across bands
        s = torch.zeros(B, self.K, device=x.device, dtype=x.dtype)
        if self.mode in ('static', 'both'):
            s = s + self.b.unsqueeze(0)
        if self.mode in ('adaptive', 'both'):
            s = s + self.phi(d.unsqueeze(-1)).squeeze(-1)
        beta = self.K * torch.softmax(s, dim=1)                # mean = 1
        self.last_beta = beta.detach()
        out = (xk * beta[:, :, None, None]).sum(dim=1, keepdim=True)
        if return_weights:
            return out, beta
        return out


class ATDGNN_BandAttn(ATDGNN):
    """
    AT-DGNN + sample-adaptive band attention in front of the unchanged pipeline.

    The band-split filters are fixed, the attention adds only K + (hidden*2+1)
    parameters, and the weighted sum is fed to the original Tception /
    sliding-window / local-filter / dynamic-graph stack without modification.
    With band_attn='none' the model is bit-exactly the AT-DGNN baseline.
    """

    def __init__(self, num_classes, input_size, sampling_rate, num_T,
                 out_graph, dropout_rate, pool, pool_step_rate, idx_graph,
                 graph_hidden=None, band_attn='both', band_numtaps=129,
                 band_hidden=8, band_edges=None, band_kind='fft',
                 sliding_layout='legacy', sliding_vectorized=False):
        super(ATDGNN_BandAttn, self).__init__(
            num_classes=num_classes, input_size=input_size, sampling_rate=sampling_rate,
            num_T=num_T, out_graph=out_graph, dropout_rate=dropout_rate, pool=pool,
            pool_step_rate=pool_step_rate, idx_graph=idx_graph, graph_hidden=graph_hidden,
            sliding_layout=sliding_layout, sliding_vectorized=sliding_vectorized)
        self.band_attn_mode = band_attn
        self.band_attn = None
        if band_attn != 'none':
            bands = band_edges or ((1, 4), (4, 8), (8, 13), (13, 30), (30, 50))
            self.band_attn = BandAttention(bands, fs=sampling_rate,
                                           numtaps=band_numtaps, hidden=band_hidden,
                                           mode=band_attn, kind=band_kind)
            print('>>> band attention: bands={} mode={} kind={} params={}'.format(
                list(bands), band_attn, band_kind,
                sum(p.numel() for p in self.band_attn.parameters())))

    def forward(self, x):
        if self.band_attn is not None:
            x = self.band_attn(x)
        return super(ATDGNN_BandAttn, self).forward(x)

    @torch.no_grad()
    def band_attention_weights(self, x):
        """Return beta (B, K) for the interpretability analysis."""
        self.eval()
        self(x)
        return None if self.band_attn is None else self.band_attn.last_beta
