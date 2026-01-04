# This file is licensed under AGPL-3.0
# Copyright (c) NXAI GmbH and its affiliates 2024
# Benedikt Alkin, Maximilian Beck, Korbinian Pöppel
import math
import warnings
from enum import Enum
import einops
import torch
import torch.nn.functional as F
from torch import nn
from vision_lstm_util import interpolate_sincos


def stochastic_depth(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    """timm-style stochastic depth on residual branch."""
    if drop_prob == 0.0 or (not training):
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    rnd = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask = torch.floor(rnd)
    return x.div(keep_prob) * mask

class SequenceConv2d(nn.Conv2d):
    """
    Applies 2D convolution to a sequence of flattened 2D patches.

    Args:
        *args: Arguments for nn.Conv2d.
        seqlens (tuple of int, optional): Spatial dimensions (height, width) of the input patches. 
                                           If None, assumes the input is square.
        **kwargs: Keyword arguments for nn.Conv2d.
    """
    def __init__(self, *args, seqlens=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.seqlens = seqlens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, channels).

        Returns:
            torch.Tensor: Output tensor after applying the convolution, reshaped back to (batch_size, seq_len, out_channels).
        """
        assert x.ndim == 3, "Input tensor must have 3 dimensions: (batch_size, seq_len, channels)."

        if self.seqlens is None:
            # Assumes square input
            h = math.sqrt(x.size(1))
            if not h.is_integer():
                raise ValueError(f"Input sequence length {x.size(1)} is not a perfect square.")
            h = int(h)
        else:
            if len(self.seqlens) != 2:
                raise ValueError("seqlens should be a tuple of length 2 (height, width).")
            h = self.seqlens[0]
        
        # Reshape input tensor from (batch_size, seq_len, channels) to (batch_size, channels, height, width)
        x = einops.rearrange(x, "b (h w) d -> b d h w", h=h)
        
        # Apply convolution
        x = super().forward(x)
        
        # Reshape output tensor back to (batch_size, seq_len, out_channels)
        x = einops.rearrange(x, "b d h w -> b (h w) d")
        
        return x

class SequenceTraversal(Enum):
    ROWWISE_FROM_TOP_LEFT = "rowwise_from_top_left"
    ROWWISE_FROM_BOT_RIGHT = "rowwise_from_bot_right"


def bias_linspace_init_(param: torch.Tensor, start: float = 3.4, end: float = 6.0) -> torch.Tensor:
    """Linearly spaced bias init across dimensions."""
    assert param.dim() == 1, f"param must be 1-dimensional (typically a bias), got {param.dim()}"
    n_dims = param.shape[0]
    init_vals = torch.linspace(start, end, n_dims)
    with torch.no_grad():
        param.copy_(init_vals)
    return param


def small_init_(param: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Fills the input Tensor with values according to the method described in Transformers without Tears: Improving
    the Normalization of Self-Attention - Nguyen, T. & Salazar, J. (2019), using a normal distribution.
    Adopted from https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py.
    """
    std = math.sqrt(2 / (5 * dim))
    torch.nn.init.normal_(param, mean=0.0, std=std)
    return param


def wang_init_(param: torch.Tensor, dim: int, num_blocks: int):
    """ Adopted from https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py. """
    std = 2 / num_blocks / math.sqrt(dim)
    torch.nn.init.normal_(param, mean=0.0, std=std)
    return param


def parallel_stabilized_simple(
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        igate_preact: torch.Tensor,
        fgate_preact: torch.Tensor,
        lower_triangular_matrix: torch.Tensor = None,
        stabilize_rowwise: bool = True,
        eps: float = 1e-6,
) -> torch.Tensor:
    """
    This is the mLSTM cell in parallel form.
    This version is stabilized. We control the range of exp() arguments by
    ensuring that they are always smaller than 0.0 by subtracting the maximum.

    Args:
        :param queries: (torch.Tensor) (B, NH, S, DH)
        :param keys: (torch.Tensor) (B, NH, S, DH)
        :param values: (torch.Tensor) (B, NH, S, DH)
        :param igate_preact: (torch.Tensor) (B, NH, S, 1)
        :param fgate_preact: (torch.Tensor) (B, NH, S, 1)
        :param lower_triangular_matrix: (torch.Tensor) (S,S). Defaults to None.
        :param stabilize_rowwise: (bool) Wether to stabilize the combination matrix C rowwise (take maximum per row).
            Alternative: Subtract the maximum over all rows. Defaults to True.
        :param eps: (float) small constant to avoid division by 0. Defaults to 1e-6.

    Returns:
        torch.Tensor: (B, NH, S, DH), h_tilde_state
    """

    orig_dtype = queries.dtype
    queries = queries.float()
    keys = keys.float()
    values = values.float()
    igate_preact = igate_preact.float()
    fgate_preact = fgate_preact.float()
    eps = 1e-6  # float32 下可以更小一点

    B, NH, S, DH = queries.shape
    _dtype, _device = queries.dtype, queries.device

    # forget gate matrix
    log_fgates = torch.nn.functional.logsigmoid(fgate_preact)  # (B, NH, S, 1)
    if lower_triangular_matrix is None or S < lower_triangular_matrix.size(-1):
        ltr = torch.tril(torch.ones((S, S), dtype=torch.bool, device=_device))
    else:
        ltr = lower_triangular_matrix
    assert ltr.dtype == torch.bool, f"lower_triangular_matrix must be of dtype bool, got {ltr.dtype}"

    log_fgates_cumsum = torch.cat(
        [
            torch.zeros((B, NH, 1, 1), dtype=_dtype, device=_device),
            torch.cumsum(log_fgates, dim=-2),
        ],
        dim=-2,
    )  # (B, NH, S+1, 1)
    # for each batch/head this is a matrix of shape (S+1, S+1) containing the cumsum of the log forget gate values
    # in the second dimension (colum dimension). Each row has the same is a copy of the first row.
    # First entry of each row is zero.
    rep_log_fgates_cumsum = log_fgates_cumsum.repeat(1, 1, 1, S + 1)  # (B, NH, S+1, S+1)
    # Now in each row cut off / subtract the forgetgate values of the later timesteps
    # where col j > row i
    _log_fg_matrix = rep_log_fgates_cumsum - rep_log_fgates_cumsum.transpose(-2, -1)  # (B, NH, S+1, S+1)
    # Causal masking & selection of the correct submatrix, such that forgetgate at timestep t is not applied
    # to the input at timestep t
    log_fg_matrix = torch.where(ltr, _log_fg_matrix[:, :, 1:, 1:], -float("inf"))  # (B, NH, S, S)

    # gate decay matrix D (combination of forget gate and input gate)
    log_D_matrix = log_fg_matrix + igate_preact.transpose(-2, -1)  # (B, NH, S, S)
    # D matrix stabilization
    if stabilize_rowwise:
        max_log_D, _ = torch.max(log_D_matrix, dim=-1, keepdim=True)  # (B, NH, S, 1)
    else:
        max_log_D = torch.max(log_D_matrix.view(B, NH, -1), dim=-1, keepdim=True)[0].unsqueeze(-1)
        # (B, NH, 1, 1)
    log_D_matrix_stabilized = log_D_matrix - max_log_D  # (B, NH, S, S)
    D_matrix = torch.exp(log_D_matrix_stabilized)  # (B, NH, S, S)

    keys_scaled = keys / math.sqrt(DH)

    # combination matrix C
    qk_matrix = queries @ keys_scaled.transpose(-2, -1)  # (B, NH, S, S)
    C_matrix = qk_matrix * D_matrix  # (B, NH, S, S)
    normalizer = torch.maximum(C_matrix.sum(dim=-1, keepdim=True).abs(), torch.exp(-max_log_D))  # (B, NH, S, 1)
    # (B, NH, S, S)
    C_matrix_normalized = C_matrix / (normalizer + eps)

    # retrieved values
    h_tilde_state = C_matrix_normalized @ values  # (B, NH, S, DH)

    h_tilde_state = h_tilde_state.to(orig_dtype)

    return h_tilde_state


class LinearHeadwiseExpand(nn.Module):
    """
    This is a structured projection layer that projects the input to a higher dimension.
    It only allows integer up-projection factors, i.e. the output dimension is a multiple of the input dimension.
    """

    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads

        dim_per_head = dim // num_heads
        self.weight = nn.Parameter(torch.empty(num_heads, dim_per_head, dim_per_head))
        if bias:
            self.bias = nn.Parameter(torch.empty(dim))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight.data, mean=0.0, std=math.sqrt(2 / 5 / self.weight.shape[-1]))
        if self.bias is not None:
            nn.init.zeros_(self.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = einops.rearrange(x, "... (nh d) -> ... nh d", nh=self.num_heads)
        x = einops.einsum(
            x,
            self.weight,
            "... nh d, nh out_d d -> ... nh out_d",
        )
        x = einops.rearrange(x, "... nh out_d -> ... (nh out_d)")
        if self.bias is not None:
            x = x + self.bias
        return x

    def extra_repr(self):
        return (
            f"dim={self.dim}, "
            f"num_heads={self.num_heads}, "
            f"bias={self.bias is not None}, "
        )


class CausalConv1d(nn.Module):
    """
    Implements causal depthwise convolution for time series data.

    Args:
        dim (int): Number of features in the input tensor (i.e., the number of input channels).
        kernel_size (int): Size of the convolution kernel. Default is 4.
        bias (bool): Whether to use a bias term in the convolution. Default is True.
        channel_mixing (bool): Whether to mix features across channels. If True, uses groups=1. If False, uses groups=dim.
    """

    def __init__(self, dim, kernel_size=4, bias=True):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.bias = bias
        # Padding ensures the output is the same length as the input
        self.pad = kernel_size - 1

        # Depthwise convolution with padding to ensure causality
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=self.pad,
            groups=dim,
            bias=bias
        )
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize convolution parameters."""
        self.conv.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the causal convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, time_steps, feature_dim).

        Returns:
            torch.Tensor: Output tensor of the same shape as input (batch_size, time_steps, feature_dim).
        """
        # Ensure input is of shape (batch_size, feature_dim, time_steps)
        x = einops.rearrange(x, 'b t f -> b f t')
        
        # Apply causal depthwise convolution
        x = self.conv(x)
        
        # Remove padding to ensure output length matches input length
        x = x[:, :, :-self.pad]
        
        # Rearrange back to (batch_size, time_steps, feature_dim)
        x = einops.rearrange(x, 'b f t -> b t f')
        
        return x

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False. """

    def __init__(
            self,
            ndim: int = -1,
            weight: bool = True,
            bias: bool = False,
            eps: float = 1e-5,
            residual_weight: bool = True,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(ndim)) if weight else None
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps
        self.residual_weight = residual_weight
        self.ndim = ndim
        self.reset_parameters()

    @property
    def weight_proxy(self) -> torch.Tensor:
        if self.weight is None:
            return None
        if self.residual_weight:
            return 1.0 + self.weight
        else:
            return self.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            normalized_shape=(self.ndim,),
            weight=self.weight_proxy,
            bias=self.bias,
            eps=self.eps,
        )

    def reset_parameters(self):
        if self.weight_proxy is not None:
            if self.residual_weight:
                nn.init.zeros_(self.weight)
            else:
                nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class MultiHeadLayerNorm(LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, "Input must be 4D tensor (B, NH, S, DH)"
        B, NH, S, DH = x.shape

        gn_in_1 = x.transpose(1, 2)  # (B, S, NH, DH)
        gn_in_2 = gn_in_1.reshape(B * S, NH * DH)  # (B * S, NH * DH)
        out = F.group_norm(
            gn_in_2,
            num_groups=NH,
            weight=self.weight_proxy,
            bias=self.bias,
            eps=self.eps,
        )  # .to(x.dtype)
        # (B * S), (NH * DH) -> (B, S, NH, DH) -> (B, NH, S, DH)
        out = out.view(B, S, NH, DH).transpose(1, 2)
        return out


class MatrixLSTMCell(nn.Module):
    def __init__(self, dim, num_heads, norm_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.igate = nn.Linear(3 * dim, num_heads)
        self.fgate = nn.Linear(3 * dim, num_heads)
        self.outnorm = MultiHeadLayerNorm(ndim=dim, weight=True, bias=norm_bias)
        self.causal_mask_cache = {}
        self.reset_parameters()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, S, _ = q.shape  # (B, S, H)

        if_gate_input = torch.cat([q, k, v], dim=-1)
        q = q.view(B, S, self.num_heads, -1)  # (B, S, NH, DH)
        k = k.view(B, S, self.num_heads, -1)  # (B, S, NH, DH)
        v = v.view(B, S, self.num_heads, -1)  # (B, S, NH, DH)

        q = q.transpose(1, 2)  # (B, NH, S, DH)
        k = k.transpose(1, 2)  # (B, NH, S, DH)
        v = v.transpose(1, 2)  # (B, NH, S, DH)

        # compute input and forget gate pre-activations
        igate_preact = self.igate(if_gate_input)  # (B, S, NH)
        igate_preact = igate_preact.transpose(-1, -2).unsqueeze(-1)  # (B, NH, S, 1)
        fgate_preact = self.fgate(if_gate_input)  # (B, S, NH)
        fgate_preact = fgate_preact.transpose(-1, -2).unsqueeze(-1)  # (B, NH, S, 1)#

        # cache causal mask to avoid memory allocation in every iteration
        key = (S, str(q.device))
        if key in self.causal_mask_cache:
            causal_mask = self.causal_mask_cache[key]
        else:
            causal_mask = torch.tril(torch.ones(S, S, dtype=torch.bool, device=q.device))
            self.causal_mask_cache[key] = causal_mask

        h_state = parallel_stabilized_simple(
            queries=q,
            keys=k,
            values=v,
            igate_preact=igate_preact,
            fgate_preact=fgate_preact,
            lower_triangular_matrix=causal_mask,
        )  # (B, NH, S, DH)

        h_state_norm = self.outnorm(h_state)  # (B, NH, S, DH)
        h_state_norm = h_state_norm.transpose(1, 2).reshape(B, S, -1)  # (B, NH, S, DH) -> (B, S, NH, DH) -> (B, S, H)

        return h_state_norm

    def reset_parameters(self):
        self.outnorm.reset_parameters()
        # forget gate initialization
        torch.nn.init.zeros_(self.fgate.weight)
        bias_linspace_init_(self.fgate.bias, start=3.0, end=6.0)
        # input gate initialization
        torch.nn.init.zeros_(self.igate.weight)
        torch.nn.init.normal_(self.igate.bias, mean=0.0, std=0.1)


class ViLLayer(nn.Module):
    def __init__(
            self,
            dim,
            direction,
            expansion=2,
            qkv_block_size=4,
            proj_bias=True,
            norm_bias=True,
            conv_bias=True,
            conv_kernel_size=4,
            conv_kind="2d",
            seqlens=None,
    ):
        super().__init__()
        assert dim % qkv_block_size == 0
        self.dim = dim
        self.direction = direction
        self.expansion = expansion
        self.qkv_block_size = qkv_block_size
        self.proj_bias = proj_bias
        self.conv_bias = conv_bias
        self.conv_kernel_size = conv_kernel_size
        self.conv_kind = conv_kind

        inner_dim = expansion * dim
        num_heads = inner_dim // qkv_block_size
        self.proj_up = nn.Linear(
            in_features=dim,
            out_features=2 * inner_dim,
            bias=proj_bias,
        )
        self.q_proj = LinearHeadwiseExpand(
            dim=inner_dim,
            num_heads=num_heads,
            bias=proj_bias,
        )
        self.k_proj = LinearHeadwiseExpand(
            dim=inner_dim,
            num_heads=num_heads,
            bias=proj_bias,
        )
        self.v_proj = LinearHeadwiseExpand(
            dim=inner_dim,
            num_heads=num_heads,
            bias=proj_bias,
        )

        if conv_kind == "causal1d":
            self.conv = CausalConv1d(
                dim=inner_dim,
                kernel_size=conv_kernel_size,
                bias=conv_bias,
            )
        elif conv_kind == "2d":
            assert conv_kernel_size % 2 == 1, \
                f"same output shape as input shape is required -> even kernel sizes not supported"
            self.conv = SequenceConv2d(
                in_channels=inner_dim,
                out_channels=inner_dim,
                kernel_size=conv_kernel_size,
                padding=conv_kernel_size // 2,
                groups=inner_dim,
                bias=conv_bias,
                seqlens=seqlens,
            )
        else:
            raise NotImplementedError
        self.mlstm_cell = MatrixLSTMCell(
            dim=inner_dim,
            num_heads=qkv_block_size,
            norm_bias=norm_bias,
        )
        self.learnable_skip = nn.Parameter(torch.ones(inner_dim))

        self.proj_down = nn.Linear(
            in_features=inner_dim,
            out_features=dim,
            bias=proj_bias,
        )
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape

        # alternate direction in successive layers
        if self.direction == SequenceTraversal.ROWWISE_FROM_TOP_LEFT:
            pass
        elif self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            x = x.flip(dims=[1])
        else:
            raise NotImplementedError

        # up-projection
        x_inner = self.proj_up(x)
        x_mlstm, z = torch.chunk(x_inner, chunks=2, dim=-1)

        # mlstm branch
        x_mlstm_conv = self.conv(x_mlstm)
        x_mlstm_conv_act = F.silu(x_mlstm_conv)
        q = self.q_proj(x_mlstm_conv_act)
        k = self.k_proj(x_mlstm_conv_act)
        v = self.v_proj(x_mlstm)
        h_tilde_state = self.mlstm_cell(q=q, k=k, v=v)
        h_tilde_state_skip = h_tilde_state + (self.learnable_skip * x_mlstm_conv_act)

        # output / z branch
        h_state = h_tilde_state_skip * F.silu(z)

        # down-projection
        x = self.proj_down(h_state)

        # reverse alternating flip
        if self.direction == SequenceTraversal.ROWWISE_FROM_TOP_LEFT:
            pass
        elif self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            x = x.flip(dims=[1])
        else:
            raise NotImplementedError

        return x

    def reset_parameters(self):
        # init inproj
        small_init_(self.proj_up.weight, dim=self.dim)
        if self.proj_up.bias is not None:
            nn.init.zeros_(self.proj_up.bias)
        # init outproj (original mLSTM uses num_blocks=1)
        wang_init_(self.proj_down.weight, dim=self.dim, num_blocks=1)
        if self.proj_down.bias is not None:
            nn.init.zeros_(self.proj_down.bias)

        nn.init.ones_(self.learnable_skip)

        def _init_qkv_proj(qkv_proj: LinearHeadwiseExpand):
            # use the embedding dim instead of the inner embedding dim
            small_init_(qkv_proj.weight, dim=self.dim)
            if qkv_proj.bias is not None:
                nn.init.zeros_(qkv_proj.bias)

        _init_qkv_proj(self.q_proj)
        _init_qkv_proj(self.k_proj)
        _init_qkv_proj(self.v_proj)

        self.mlstm_cell.reset_parameters()

class ViLBlock(nn.Module):
    def __init__(
        self,
        dim,
        direction,
        drop_path=0.0,
        conv_kind="2d",
        conv_kernel_size=3,
        proj_bias=True,
        norm_bias=True,
        seqlens=None,
        gamma_init: float = 1e-4,   # ✅ 可扫 1e-6/1e-5/1e-4
    ):
        super().__init__()
        self.dim = dim
        self.direction = direction
        self.drop_prob = float(drop_path)

        self.norm = LayerNorm(ndim=dim, weight=True, bias=norm_bias, eps=1e-6)
        self.layer = ViLLayer(
            dim=dim,
            direction=direction,
            conv_kind=conv_kind,
            conv_kernel_size=conv_kernel_size,
            seqlens=seqlens,
            norm_bias=norm_bias,
            proj_bias=proj_bias,
        )
        self.gamma = nn.Parameter(torch.ones(dim) * gamma_init)
        self.reset_parameters()

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        # residual branch only
        y = self.layer(self.norm(x))
        return self.gamma * y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ✅ 强制 residual add，不依赖 util.DropPath 的实现细节
        d = self.delta(x)
        d = stochastic_depth(d, self.drop_prob, self.training)
        return x + d

    def reset_parameters(self):
        self.layer.reset_parameters()
        self.norm.reset_parameters()


def to_ntuple(x, n):
    """ Convert a scalar or tuple to a tuple of length `n`. """
    if isinstance(x, tuple):
        assert len(x) == n, f"Expected tuple of length {n}, got {len(x)}"
        return x
    return (x,) * n


class VitPatchEmbed(nn.Module):
    def __init__(self, dim, num_channels, resolution, patch_size, stride=None, init_weights="xavier_uniform"):
        """
        Args:
            dim (int): Dimensionality of the output embeddings.
            num_channels (int): Number of input channels.
            resolution (tuple): Spatial resolution of the input image.
            patch_size (int or tuple): Size of each patch.
            stride (int or tuple, optional): Stride of the convolutional layer. Defaults to patch_size.
            init_weights (str, optional): Weight initialization method. Options are "xavier_uniform" or "torch". Defaults to "xavier_uniform".
        """
        super().__init__()
        self.resolution = resolution
        self.init_weights = init_weights
        self.ndim = len(resolution)
        self.patch_size = to_ntuple(patch_size, n=self.ndim)
        self.stride = to_ntuple(stride, n=self.ndim) if stride is not None else self.patch_size

        # Validate resolution and patch size
        for i in range(self.ndim):
            assert (resolution[i] - self.patch_size[i]) % self.stride[i] == 0, \
                f"Bad (resolution, patch, stride) at dim {i}: {resolution[i]}, {self.patch_size[i]}, {self.stride[i]}"

        self.seqlens = [
            (resolution[i] - self.patch_size[i]) // self.stride[i] + 1
            for i in range(self.ndim)
        ]
        self.num_patches = int(self.seqlens[0] * self.seqlens[1])

        # Choose appropriate convolution function
        if self.ndim == 1:
            conv_ctor = nn.Conv1d
        elif self.ndim == 2:
            conv_ctor = nn.Conv2d
        elif self.ndim == 3:
            conv_ctor = nn.Conv3d
        else:
            raise NotImplementedError("Dimension not supported.")

        self.proj = conv_ctor(num_channels, dim, kernel_size=self.patch_size, stride=self.stride)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights based on the specified method."""
        if self.init_weights == "torch":
            pass  # Default initialization
        elif self.init_weights == "xavier_uniform":
            w = self.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.zeros_(self.proj.bias)
        else:
            raise NotImplementedError("Initialization method not supported.")

    def forward(self, x):
        #print("PatchEmbed input shape:", x.shape)
        """
        Forward pass of the module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_channels, H, W) or (batch_size, num_channels, D, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_patches, dim)
        """
        for i in range(self.ndim):
            s = x.size(i + 2)
            assert s >= self.patch_size[i] and (s - self.patch_size[i]) % self.stride[i] == 0, \
                f"x.shape={x.shape} incompatible with patch={self.patch_size} stride={self.stride}"

        
        # Apply convolution to extract patches and project them
        x = self.proj(x)
        
        # Rearrange tensor from (batch_size, dim, H', W') to (batch_size, num_patches, dim)
        x = einops.rearrange(x, "b c ... -> b ... c")
        
        return x


class VitPosEmbed2d(nn.Module):
    def __init__(self, seqlens, dim: int, allow_interpolation: bool = True):
        """
        Args:
            seqlens (tuple): Sequence lengths for each spatial dimension (height, width).
            dim (int): Dimensionality of the positional embeddings.
            allow_interpolation (bool): Whether to allow interpolation of positional embeddings.
        """
        super().__init__()
        self.seqlens = seqlens
        self.dim = dim
        self.allow_interpolation = allow_interpolation
        self.embed = nn.Parameter(torch.zeros(1, *seqlens, dim))
        self.reset_parameters()

    @property
    def _expected_x_ndim(self):
        return len(self.seqlens) + 2

    def reset_parameters(self):
        """Initialize positional embeddings with truncated normal distribution."""
        nn.init.trunc_normal_(self.embed, std=.02)

    def forward(self, x):
        """
        Forward pass of the module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_patches, dim) or similar.

        Returns:
            torch.Tensor: Output tensor with added positional embeddings.
        """
        assert x.ndim == self._expected_x_ndim, f"Expected input with {self._expected_x_ndim} dimensions, got {x.ndim}"
        
        if x.shape[1:] != self.embed.shape[1:]:
            if not self.allow_interpolation:
                raise ValueError("Shape mismatch and interpolation not allowed.")
            # Interpolation function should be defined elsewhere
            embed = interpolate_sincos(embed=self.embed, seqlens=x.shape[1:-1])
        else:
            embed = self.embed
        
        return x + embed

class ViLBlockPair(nn.Module):
    def __init__(
        self,
        dim,
        drop_path=0.0,
        conv_kind="2d",
        conv_kernel_size=3,
        proj_bias=True,
        norm_bias=True,
        seqlens=None,
        gamma_init: float = 1e-4,
        fusion: str = "serial",   # ✅ serial | parallel_add | parallel_gated | parallel_concat
    ):
        super().__init__()
        self.dim = dim
        self.drop_prob = float(drop_path)
        self.fusion = fusion.lower()

        self.rowwise_from_top_left = ViLBlock(
            dim=dim,
            direction=SequenceTraversal.ROWWISE_FROM_TOP_LEFT,
            drop_path=drop_path,
            conv_kind=conv_kind,
            conv_kernel_size=conv_kernel_size,
            proj_bias=proj_bias,
            norm_bias=norm_bias,
            seqlens=seqlens,
            gamma_init=gamma_init,
        )
        self.rowwise_from_bot_right = ViLBlock(
            dim=dim,
            direction=SequenceTraversal.ROWWISE_FROM_BOT_RIGHT,
            drop_path=drop_path,
            conv_kind=conv_kind,
            conv_kernel_size=conv_kernel_size,
            proj_bias=proj_bias,
            norm_bias=norm_bias,
            seqlens=seqlens,
            gamma_init=gamma_init,
        )

        if self.fusion == "parallel_gated":
            # gate 用全局 token mean -> (B, dim)，更稳更省
            self.gate_norm = LayerNorm(ndim=dim, weight=True, bias=norm_bias, eps=1e-6)
            self.gate = nn.Linear(dim, dim, bias=True)
        elif self.fusion == "parallel_concat":
            self.fuse_norm = LayerNorm(ndim=2 * dim, weight=True, bias=norm_bias, eps=1e-6)
            self.fuse = nn.Linear(2 * dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fusion == "serial":
            x = self.rowwise_from_top_left(x)
            x = self.rowwise_from_bot_right(x)
            return x

        # ✅ 并行：两个方向都用同一个输入 x 计算 delta，再融合后一次性 residual add
        d1 = self.rowwise_from_top_left.delta(x)
        d2 = self.rowwise_from_bot_right.delta(x)

        if self.fusion == "parallel_add":
            d = d1 + d2

        elif self.fusion == "parallel_gated":
            g = self.gate(self.gate_norm(x).mean(dim=1))   # (B, dim)
            g = torch.sigmoid(g).unsqueeze(1)              # (B,1,dim)
            d = g * d1 + (1.0 - g) * d2

        elif self.fusion == "parallel_concat":
            d = torch.cat([d1, d2], dim=-1)                # (B,S,2D)
            d = self.fuse(self.fuse_norm(d))               # (B,S,D)

        else:
            raise ValueError(f"unknown fusion='{self.fusion}'")

        d = stochastic_depth(d, self.drop_prob, self.training)
        return x + d

class ColViLBlockPair(nn.Module):
    """
    Column-wise modeling via H/W transpose:
      row-major (H,W) -> transpose -> row-major (W,H) -> ViLBlockPair(swapped seqlens) -> transpose back
    Works with conv_kind="2d" because internal seqlens are swapped to match layout.
    """
    def __init__(
        self,
        dim,
        drop_path=0.0,
        conv_kind="2d",
        conv_kernel_size=3,
        proj_bias=True,
        norm_bias=True,
        seqlens=None,
        gamma_init: float = 1e-4,
        fusion: str = "serial",
    ):
        super().__init__()
        assert seqlens is not None and len(seqlens) == 2
        self.in_seqlens = tuple(seqlens)
        H, W = self.in_seqlens
        self.inner = ViLBlockPair(
            dim=dim,
            drop_path=drop_path,
            conv_kind=conv_kind,
            conv_kernel_size=conv_kernel_size,
            proj_bias=proj_bias,
            norm_bias=norm_bias,
            seqlens=(W, H),           # ✅ swapped
            gamma_init=gamma_init,
            fusion=fusion,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        H, W = self.in_seqlens
        assert N == H * W, f"expected N=H*W={H*W}, got {N}"

        # (B, H, W, D) -> (B, W, H, D) -> flatten as (W,H)
        x_hw = einops.rearrange(x, "b (h w) d -> b h w d", h=H, w=W).transpose(1, 2)
        x2 = einops.rearrange(x_hw, "b h w d -> b (h w) d")
        x2 = self.inner(x2)

        # back: (B, W, H, D) -> (B, H, W, D) -> flatten as (H,W)
        y_hw = einops.rearrange(x2, "b (h w) d -> b h w d", h=W, w=H).transpose(1, 2)
        y = einops.rearrange(y_hw, "b h w d -> b (h w) d")
        return y


class HaarDWT2d(nn.Module):
    """
    Fixed-weight 2D Haar DWT（depthwise conv + stride=2），输出四个子带：LL, LH, HL, HH。
    GPU 友好且可微（权重不训练）。
    """
    def __init__(self, channels: int, padding: str = "reflect"):
        super().__init__()
        self.channels = channels
        self.padding = padding
        h = torch.tensor([1.0, 1.0]) / (2.0 ** 0.5)
        g = torch.tensor([1.0, -1.0]) / (2.0 ** 0.5)
        LL = torch.einsum('i,j->ij', h, h)  # 2x2
        LH = torch.einsum('i,j->ij', h, g)
        HL = torch.einsum('i,j->ij', g, h)
        HH = torch.einsum('i,j->ij', g, g)
        weight = torch.zeros(4 * channels, 1, 2, 2)
        for c in range(channels):
            weight[c*4 + 0, 0] = LL
            weight[c*4 + 1, 0] = LH
            weight[c*4 + 2, 0] = HL
            weight[c*4 + 3, 0] = HH
        self.register_buffer('weight', weight)

    def forward(self, x):  # x: (B,C,H,W)
        B, C, H, W = x.shape
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            # 只补右/下，避免整体偏移
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        y = F.conv2d(x, self.weight, stride=2, padding=0, groups=self.channels)

        B, OC, H2, W2 = y.shape
        C = self.channels
        y = y.view(B, C, 4, H2, W2)
        LL, LH, HL, HH = y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3]  # (B,C,H/2,W/2)
        return LL, LH, HL, HH

class HaarTokenMerging(nn.Module):
    """
    Token-level downsample via fixed Haar DWT:
      (B, H*W, C) -> reshape -> (B, C, H, W)
      HaarDWT2d -> concat 4 subbands -> (B, 4C, H/2, W/2)
      LN + Linear(4C -> C_out) -> flatten -> (B, (H/2)*(W/2), C_out)
    """
    def __init__(self, in_dim: int, out_dim: int, in_seqlens: tuple, norm_bias: bool = True):
        super().__init__()
        assert len(in_seqlens) == 2
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_seqlens = tuple(in_seqlens)

        H, W = self.in_seqlens
        self.out_seqlens = ((H + 1) // 2, (W + 1) // 2)  # reflect-pad 后 stride=2 的等价输出

        self.haar = HaarDWT2d(in_dim)
        self.norm = LayerNorm(ndim=4 * in_dim, weight=True, bias=norm_bias, eps=1e-6)
        self.reduction = nn.Linear(4 * in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        B, N, C = x.shape
        H, W = self.in_seqlens
        assert C == self.in_dim, f"expected C={self.in_dim}, got {C}"
        assert N == H * W, f"expected N=H*W={H*W}, got N={N}"

        x = einops.rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        ll, lh, hl, hh = self.haar(x)                         # each: (B, C, H2, W2)
        y = torch.cat([ll, lh, hl, hh], dim=1)                # (B, 4C, H2, W2)

        y = einops.rearrange(y, "b c h w -> b h w c")         # (B, H2, W2, 4C)
        y = self.norm(y)
        y = self.reduction(y)                                  # (B, H2, W2, out_dim)
        y = einops.rearrange(y, "b h w c -> b (h w) c")        # (B, H2*W2, out_dim)
        return y

# === 旁路→主干 的残差适配器 ===
class HeadResidualAdapter(nn.Module):
    """
    把分支向量投到 head_dim ，以可学习缩放的方式残差注入到主干向量里。
    """
    def __init__(self, head_dim: int, branch_dim: int, init_scale: float = 1e-2):
        super().__init__()
        self.proj = nn.Linear(branch_dim, head_dim, bias=True)
        self.alpha = nn.Parameter(torch.ones(head_dim) * init_scale)  # LayerScale 风格
    def forward(self, main_vec: torch.Tensor, branch_vec: torch.Tensor) -> torch.Tensor:
        delta = self.proj(branch_vec)
        return main_vec + self.alpha * delta


# === 残差卷积块 ===
class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.same = (in_ch == out_ch and s == 1)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.act1 = nn.SiLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, k, stride=1, padding=p, bias=False)
        self.short = nn.Identity() if self.same else nn.Conv2d(in_ch, out_ch, 1, stride=s, bias=False)
        self.gamma = nn.Parameter(torch.ones(1) * 1e-4)  # LayerScale：初始很小，先“别叠太多改动”

    def forward(self, x):
        identity = self.short(x)
        y = self.conv1(self.act1(self.bn1(x)))
        y = self.conv2(self.act2(self.bn2(y)))
        return identity + self.gamma * y


# === : 轻量局部残差混合器（DWConv）===
class ResidualDepthwiseMix(nn.Module):
    """
    在 token 序列与网格之间做一次极轻的 3x3 深度可分离卷积，再残差回去。
    grid = (H_patches) == (W_patches) ，CIFAR-10 且 patch=4 时一般是 8 或 4。
    """
    def __init__(self, d_model: int, grid: int):
        super().__init__()
        self.grid = grid
        self.dw = nn.Conv2d(d_model, d_model, 3, padding=1, groups=d_model, bias=False)
        self.pw = nn.Conv2d(d_model, d_model, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):  # x: (B, L, D)
        B, L, D = x.shape
        g = self.grid
        x2 = x.transpose(1, 2).reshape(B, D, g, g)
        y = self.pw(self.dw(x2)).reshape(B, D, g*g).transpose(1, 2)
        return x + self.gamma * y

class FeatureExtractor(nn.Module):
    """
    先可选 DWT（输出 LL/LH/HL/HH），再用轻量卷积堆叠提取局部特征。
    dwt_fuse: 'LL' | 'concat' | 'add' | 'gated'
    """
    def __init__(self, input_channels: int, conv_channels: list,
                 use_dwt: bool = False, dwt_fuse: str = "LL"):
        super().__init__()
        self.use_dwt = bool(use_dwt)
        self.dwt_fuse = dwt_fuse
        C = input_channels

        self.dwt = None
        self.reduce = None
        self.hf_reduce = None
        self.hf_gate = None

        if self.use_dwt:
            self.dwt = HaarDWT2d(C)

            if dwt_fuse == "LL":
                first_in = C

            elif dwt_fuse == "concat":
                first_in = 4 * C

            elif dwt_fuse == "add":
                first_in = C
                # 用 4C->C，把 LL/LH/HL/HH 综合成一个 C 通道的残差项
                self.reduce = nn.Conv2d(4 * C, C, kernel_size=1, bias=False)

            elif dwt_fuse == "gated":
                first_in = C
                # 高频 3C -> C
                self.hf_reduce = nn.Conv2d(3 * C, C, kernel_size=1, bias=True)
                # gate: [ll(C), hf3(C)] -> gate(C)
                self.hf_gate = nn.Sequential(
                    nn.Conv2d(2 * C, C, kernel_size=1, bias=True),
                    nn.Sigmoid()
                )

            else:
                raise ValueError("dwt_fuse must be one of {'LL','concat','add','gated'}")
        else:
            first_in = C

        blocks = []
        in_ch = first_in
        for out_ch in conv_channels:
            blocks.append(ResidualConvBlock(in_ch, out_ch, k=3, s=1, p=1))
            in_ch = out_ch
        self.conv_features = nn.Sequential(*blocks)
        self.final_channels = conv_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_dwt:
            z = x
        else:
            ll, lh, hl, hh = self.dwt(x)

            if self.dwt_fuse == "LL":
                z = ll

            elif self.dwt_fuse == "concat":
                z = torch.cat([ll, lh, hl, hh], dim=1)

            elif self.dwt_fuse == "add":
                all4 = torch.cat([ll, lh, hl, hh], dim=1)      # 4C
                z = ll + self.reduce(all4)                     # C

            elif self.dwt_fuse == "gated":
                hf = torch.cat([lh, hl, hh], dim=1)            # 3C
                hf3 = self.hf_reduce(hf)                       # C
                gate = self.hf_gate(torch.cat([ll, hf3], dim=1))  # C
                z = ll + gate * hf3

            else:
                z = ll  # 保底

        return self.conv_features(z)

class VisionLSTM2(nn.Module):
    def __init__(
        self,
        dim, input_shape, patch_size, depth, output_shape, mode, pooling,
        drop_path_rate, drop_path_decay, stride, legacy_norm, conv_kind,
        conv_kernel_size, proj_bias, norm_bias,
        feature_extractor_channels, use_dwt=False,

        # ===== 新增：金字塔开关 =====
        pyramid: str = "none",         # "none" | "half" | "half2" | "full"
        stage_dims=None,               # 允许你手工传 [d1,d2,...,dim]，不传就自动算
        stage_depths=None,             # 允许你手工传 [n1,n2,...]，不传就自动切
        mixer_every: int = 2,

        # ===== 新增：ViLBlockPair/ColViLBlockPair 的控制 =====
        pair_fusion: str = "parallel_gated",   # "serial" | "parallel_add" | "parallel_gated" | "parallel_concat"
        col_every: int = 0,                    # 0=关闭；比如 2 表示每 2 个 block 用一次 ColViLBlockPair
        gamma_init: float = 1e-4,              # 建议扫 1e-6/1e-5/1e-4

    ):
        super().__init__()

        self.dim = dim                      # 这里 dim 仍作为最终 head dim（final dim）
        self.input_shape = input_shape
        self.patch_size = patch_size
        self.depth = depth
        self.output_shape = output_shape
        self.mode = mode
        self.pooling = pooling
        self.drop_path_rate = drop_path_rate
        self.drop_path_decay = drop_path_decay
        self.stride = stride
        self.legacy_norm = legacy_norm
        self.conv_kind = conv_kind
        self.conv_kernel_size = conv_kernel_size
        self.proj_bias = proj_bias
        self.norm_bias = norm_bias
        self.mixer_every = mixer_every
        self.pair_fusion = pair_fusion
        self.col_every = int(col_every)
        self.gamma_init = float(gamma_init)


        # -------------------------
        # 0) Feature extractor (你的原逻辑保留)
        # -------------------------
        self.feature_extractor = FeatureExtractor(
            input_channels=input_shape[0],
            conv_channels=feature_extractor_channels,
            use_dwt=use_dwt,
            #dwt_fuse='gated',  # 你现在用 gated；如果想更稳建议改回 'LL'
            dwt_fuse='LL',
        )

        num_channels = feature_extractor_channels[-1]
        pe_res = (input_shape[1] // 2, input_shape[2] // 2) if use_dwt else (input_shape[1], input_shape[2])

        # -------------------------
        # 1) 解析金字塔配置
        # -------------------------
        def _ceil_to(x, m=8):
            x = int(x)
            return max(m, ((x + m - 1) // m) * m)

        pyramid = pyramid.lower()
        if stage_dims is None:
            if pyramid == "none":
                stage_dims = [dim]
            elif pyramid == "half":
                stage_dims = [_ceil_to(dim // 2), dim]
            elif pyramid == "half2":
                stage_dims = [_ceil_to(dim // 4), _ceil_to(dim // 2), dim]
            elif pyramid == "full":
                stage_dims = [_ceil_to(dim // 8), _ceil_to(dim // 4), _ceil_to(dim // 2), dim]
            else:
                raise ValueError(f"unknown pyramid='{pyramid}'")
        else:
            stage_dims = list(stage_dims)
            assert stage_dims[-1] == dim, "stage_dims 的最后一个必须等于 dim（final dim）"

        num_stages = len(stage_dims)
        num_merges = num_stages - 1

        if stage_depths is None:
            # 默认：越往后（分辨率越低）block 越多
            if num_stages == 1:
                stage_depths = [depth]
            elif num_stages == 2:
                d1 = max(1, depth // 3)
                stage_depths = [d1, depth - d1]
            elif num_stages == 3:
                d1 = max(1, depth // 6)
                d2 = max(1, depth // 3)
                stage_depths = [d1, d2, depth - d1 - d2]
            elif num_stages == 4:
                d1 = max(1, depth // 10)
                d2 = max(1, depth // 10)
                d3 = max(1, depth * 6 // 10)
                stage_depths = [d1, d2, d3, depth - d1 - d2 - d3]
            else:
                raise ValueError("num_stages too large; please pass stage_depths explicitly")
        else:
            stage_depths = list(stage_depths)
            assert sum(stage_depths) == depth, "stage_depths 之和必须等于 depth"

        assert len(stage_depths) == num_stages

        self.stage_dims = stage_dims
        self.stage_depths = stage_depths

        # -------------------------
        # 2) PatchEmbed 输出 dim 改为 stage_dims[0]（stem dim）
        # -------------------------
        stem_dim = stage_dims[0]
        self.patch_embed = VitPatchEmbed(
            dim=stem_dim,
            num_channels=num_channels,
            resolution=pe_res,
            patch_size=patch_size,
            stride=stride,
            init_weights="xavier_uniform",
        )

        # PosEmbed 只在 stem 分辨率加一次（最小改动版本）
        self.pos_embed = VitPosEmbed2d(seqlens=self.patch_embed.seqlens, dim=stem_dim)

        # -------------------------
        # 3) DropPath schedule（按总 depth 线性分配）
        # -------------------------
        if drop_path_decay and drop_path_rate > 0.:
            dpr_all = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        else:
            dpr_all = [drop_path_rate] * depth

        # -------------------------
        # 4) 构建 stages + merges（核心改动）
        # -------------------------
        self.stage_blocks = nn.ModuleList()
        self.stage_mixers = nn.ModuleList()
        self.merges = nn.ModuleList()

        seqlens = tuple(self.patch_embed.seqlens)
        self.stage_seqlens = [seqlens]

        dpr_cursor = 0
        for si in range(num_stages):
            sd = stage_dims[si]
            sl = self.stage_seqlens[si]
            sdepth = stage_depths[si]

            # blocks
            blocks = nn.ModuleList()
            for bi in range(sdepth):
                use_col = (self.col_every > 0) and ((bi + 1) % self.col_every == 0)
                BlockCtor = ColViLBlockPair if use_col else ViLBlockPair
                blocks.append(
                    BlockCtor(
                        dim=sd,
                        drop_path=dpr_all[dpr_cursor + bi],
                        conv_kind=conv_kind,
                        conv_kernel_size=conv_kernel_size,
                        seqlens=sl,
                        proj_bias=proj_bias,
                        norm_bias=norm_bias,
                        gamma_init=self.gamma_init,
                        fusion=self.pair_fusion,
                    )
                )

            dpr_cursor += sdepth
            self.stage_blocks.append(blocks)

            # mixers（每个 stage 用自己的 grid）
            gh, gw = sl
            assert gh == gw, f"ResidualDepthwiseMix 目前要求方形 grid，但 got {sl}"
            nmix = (sdepth + self.mixer_every - 1) // self.mixer_every
            mixers = nn.ModuleList([ResidualDepthwiseMix(sd, grid=gh) for _ in range(nmix)])
            self.stage_mixers.append(mixers)

            # merge（stage i -> i+1）
            if si < num_merges:
                out_dim = stage_dims[si + 1]
                merge = HaarTokenMerging(in_dim=sd, out_dim=out_dim, in_seqlens=sl, norm_bias=norm_bias)
                self.merges.append(merge)
                self.stage_seqlens.append(merge.out_seqlens)

        self.final_seqlens = self.stage_seqlens[-1]
        self.final_dim = stage_dims[-1]

        # -------------------------
        # 5) Norm / Head（维度改为 final_dim==dim）
        # -------------------------
        if pooling == "bilateral_flatten":
            head_dim = self.final_dim * 2
        else:
            head_dim = self.final_dim

        self.norm = LayerNorm(self.final_dim, bias=norm_bias, eps=1e-6)
        self.legacy_norm = nn.LayerNorm(head_dim) if legacy_norm else nn.Identity()

        if mode == "features":
            assert self.output_shape is None
            self.head = None
            self.head_adapter = None
            if self.pooling is None:
                # 输出 token features: (N_final, dim_final)
                self.output_shape = (self.final_seqlens[0] * self.final_seqlens[1], self.final_dim)
            elif self.pooling == "to_image":
                self.output_shape = (self.final_dim, *self.final_seqlens)
            else:
                raise NotImplementedError(f"invalid pooling '{pooling}' for mode '{mode}'")
        elif mode == "classifier":
            assert self.output_shape is not None and len(self.output_shape) == 1, \
                "output_shape=(num_classes,) required in classifier mode"
            self.head = nn.Linear(head_dim, self.output_shape[0])
            nn.init.trunc_normal_(self.head.weight, std=2e-5)
            nn.init.zeros_(self.head.bias)
            self.head_adapter = HeadResidualAdapter(head_dim=head_dim, branch_dim=self.final_dim)

            # ✅ 新增：分支-主干融合门控（标量）
            self.gate_layer = nn.Sequential(
                nn.Linear(head_dim * 2, head_dim),
                nn.ReLU(inplace=True),
                nn.Linear(head_dim, 1),
                nn.Sigmoid()
            )

            # （可选但很推荐）让初始 gate 偏向“少注入”，更稳
            nn.init.zeros_(self.gate_layer[2].weight)
            nn.init.constant_(self.gate_layer[2].bias, -2.0)  # sigmoid(-2)=0.119
        else:
            raise NotImplementedError

        # -------------------------
        # 6) 分支保持：输出 dim 需要等于 final_dim（也就是 dim）
        # -------------------------
        branch_in = feature_extractor_channels[-1]
        self.feature_extractor_branch = nn.Sequential(
            nn.Conv2d(in_channels=branch_in, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(8),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 384),
            nn.SiLU(),
            nn.Linear(384, self.final_dim),
        )

    def load_state_dict(self, state_dict, strict=True):
        # 只对 stem 的 pos_embed 做插值兼容
        if "pos_embed.embed" in state_dict:
            old_pos_embed = state_dict["pos_embed.embed"]
            if old_pos_embed.shape != self.pos_embed.embed.shape:
                state_dict["pos_embed.embed"] = interpolate_sincos(embed=old_pos_embed, seqlens=self.pos_embed.seqlens)
        return super().load_state_dict(state_dict=state_dict, strict=strict)

    @torch.jit.ignore
    def no_weight_decay(self):
        # 更稳一点：把 norm / gamma / alpha / learnable_skip / bias 都排除（可按你训练脚本需要调整）
        skip = {"pos_embed.embed"}
        for n, _p in self.named_parameters():
            if n.endswith(".bias"):
                skip.add(n)
            if ".norm." in n or "legacy_norm" in n:
                skip.add(n)
            if "gamma" in n or "alpha" in n or "learnable_skip" in n:
                skip.add(n)
        return skip

    def _forward_one_stage(self, x, blocks: nn.ModuleList, mixers: nn.ModuleList):
        midx = 0
        for i, block in enumerate(blocks):
            x = block(x)
            if (i % self.mixer_every) == 0:
                x = mixers[midx](x)
                midx += 1
        return x

    def forward(self, x):
        # ----- 主干 -----
        feature_maps = self.feature_extractor(x)

        # stem tokens: (B, gh, gw, stem_dim)
        x_main = self.patch_embed(feature_maps)
        x_main = self.pos_embed(x_main)
        x_main = einops.rearrange(x_main, "b h w d -> b (h w) d")

        # stages + merges
        for si in range(len(self.stage_blocks)):
            x_main = self._forward_one_stage(x_main, self.stage_blocks[si], self.stage_mixers[si])
            if si < len(self.merges):
                x_main = self.merges[si](x_main)

        x_main = self.norm(x_main)

        # pooling
        if self.pooling is None:
            x_main = self.legacy_norm(x_main)
        elif self.pooling == "to_image":
            x_main = self.legacy_norm(x_main)
            seqlen_h, seqlen_w = self.final_seqlens
            x_main = einops.rearrange(
                x_main,
                "b (seqlen_h seqlen_w) dim -> b dim seqlen_h seqlen_w",
                seqlen_h=seqlen_h,
                seqlen_w=seqlen_w,
            )
        elif self.pooling == "global":
            x_main = x_main.mean(dim=1)
            x_main = self.legacy_norm(x_main)
        elif self.pooling == "bilateral_avg":
            x_main = (x_main[:, 0] + x_main[:, -1]) / 2
            x_main = self.legacy_norm(x_main)
        elif self.pooling == "bilateral_flatten":
            x_main = torch.concat([x_main[:, 0], x_main[:, -1]], dim=1)
            x_main = self.legacy_norm(x_main)
        else:
            raise NotImplementedError(f"pooling '{self.pooling}' is not implemented")

        # ----- 分支 -----
        feature_branch_out = self.feature_extractor_branch(feature_maps)  # (B, final_dim)

        # classifier head
        if self.mode == "classifier":
            # 先把分支投到 head_dim
            branch_proj = self.head_adapter.proj(feature_branch_out)  # (B, head_dim)

            # ✅ 标量门控：输入 [主干, 分支投影] → g
            g = self.gate_layer(torch.cat([x_main, branch_proj], dim=-1))  # (B, 1)

            # ✅ 注入：x + g * (alpha ⊙ branch_proj)
            x_main = x_main + g * (self.head_adapter.alpha * branch_proj)

            return self.head(x_main)

        # features mode
        return x_main
