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
from torchvision.models import convnext_tiny
from vision_lstm_util import interpolate_sincos, to_ntuple, DropPath
import torch.fft

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
        if S in self.causal_mask_cache:
            causal_mask = self.causal_mask_cache[(S, str(q.device))]
        else:
            causal_mask = torch.tril(torch.ones(S, S, dtype=torch.bool, device=q.device))
            self.causal_mask_cache[(S, str(q.device))] = causal_mask

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
    ):
        super().__init__()
        self.dim = dim
        self.direction = direction
        self.drop_path = drop_path
        self.conv_kind = conv_kind
        self.conv_kernel_size = conv_kernel_size

        self.drop_path = DropPath(drop_prob=drop_path)
        self.norm = LayerNorm(ndim=dim, weight=True, bias=norm_bias)
        self.layer = ViLLayer(
            dim=dim,
            direction=direction,
            conv_kind=conv_kind,
            conv_kernel_size=conv_kernel_size,
            seqlens=seqlens,
            norm_bias=norm_bias,
            proj_bias=proj_bias,
        )
        #self.gamma = nn.Parameter(torch.ones(dim) * 1e-2)
        self.gamma = nn.Parameter(torch.ones(dim) * 1e-4)
        self.reset_parameters()

    def _forward_path(self, x):
        x = self.norm(x)
        x = self.layer(x)
        #return x
        return self.gamma * x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop_path(x, self._forward_path)
        return x

    def reset_parameters(self):
        self.layer.reset_parameters()
        self.norm.reset_parameters()


def to_ntuple(x, n):
    """ Convert a scalar or tuple to a tuple of length `n`. """
    if isinstance(x, tuple):
        assert len(x) == n, f"Expected tuple of length {n}, got {len(x)}"
        return x
    return (x,) * n



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
            assert resolution[i] % self.patch_size[i] == 0, \
                f"Resolution[{i}] % Patch_Size[{i}] != 0 (Resolution={resolution}, Patch_Size={patch_size})"

        self.seqlens = [resolution[i] // self.patch_size[i] for i in range(self.ndim)]

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
        assert all(x.size(i + 2) % self.patch_size[i] == 0 for i in range(self.ndim)), \
            f"x.shape={x.shape} incompatible with patch_size={self.patch_size}"
        
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
    ):
        super().__init__()
        self.rowwise_from_top_left = ViLBlock(
            dim=dim,
            direction=SequenceTraversal.ROWWISE_FROM_TOP_LEFT,
            drop_path=drop_path,
            conv_kind=conv_kind,
            conv_kernel_size=conv_kernel_size,
            proj_bias=proj_bias,
            norm_bias=norm_bias,
            seqlens=seqlens
        )
        self.rowwise_from_bot_right = ViLBlock(
            dim=dim,
            direction=SequenceTraversal.ROWWISE_FROM_BOT_RIGHT,
            drop_path=drop_path,
            conv_kind=conv_kind,
            conv_kernel_size=conv_kernel_size,
            proj_bias=proj_bias,
            norm_bias=norm_bias,
            seqlens=seqlens
        )

    def forward(self, x):
        x = self.rowwise_from_top_left(x)
        x = self.rowwise_from_bot_right(x)
        return x



class FourierTransform(nn.Module):
    def __init__(self):
        super(FourierTransform, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the 2D Fourier transform
        x_fft = torch.fft.fft2(x)
        # Compute the magnitude of the complex result
        x_fft_magnitude = torch.abs(x_fft)
        # Normalize
        x_fft_normalized = torch.log1p(x_fft_magnitude)
        return x_fft_normalized

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
        if self.padding == 'reflect':
            x = F.pad(x, (1,0,1,0), mode='reflect')
        y = F.conv2d(x, self.weight, stride=2, padding=0, groups=self.channels)  # (B,4C,H/2,W/2)
        B, OC, H2, W2 = y.shape
        C = self.channels
        y = y.view(B, C, 4, H2, W2)
        LL, LH, HL, HH = y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3]  # (B,C,H/2,W/2)
        return LL, LH, HL, HH

# === NEW: 旁路→主干 的残差适配器 ===
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


# === NEW: 残差卷积块（替换 FeatureExtractor 里的 Conv+SiLU 堆叠）===
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
        #self.gamma = nn.Parameter(torch.zeros(1))  # LayerScale：初始很小，先“别叠太多改动”
        self.gamma = nn.Parameter(torch.zeros(1e-4))  # LayerScale：初始很小，先“别叠太多改动”

    def forward(self, x):
        identity = self.short(x)
        y = self.conv1(self.act1(self.bn1(x)))
        y = self.conv2(self.act2(self.bn2(y)))
        return identity + self.gamma * y


# === NEW（可选）: 轻量局部残差混合器（DWConv）===
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
    WaveCNet 风格：先做一次 DWT 下采样（默认只用 LL），再进入小卷积堆叠提取局部特征。
    注意：保留了原有的 use_fourier 参数名作为“启用 DWT”的别名，方便你现有调用不改动。
    """
    def __init__(self, input_channels: int, conv_channels: list, use_fourier: bool = False,
                 use_dwt: bool = None, dwt_fuse: str = 'LL'):
        super(FeatureExtractor, self).__init__()
        # 兼容：如果未显式传 use_dwt，则沿用 use_fourier 的值
        self.use_dwt = use_dwt if use_dwt is not None else use_fourier
        self.dwt_fuse = dwt_fuse
        self.input_channels = input_channels

        if self.use_dwt:
            self.dwt = HaarDWT2d(input_channels)
            if dwt_fuse == 'LL':
                first_in = input_channels
                self.reduce = None
            elif dwt_fuse == 'concat':
                first_in = input_channels * 4
                self.reduce = None
            elif dwt_fuse == 'add':
                first_in = input_channels
                self.reduce = nn.Conv2d(input_channels * 4, input_channels, kernel_size=1, bias=False)
            else:
                raise ValueError("dwt_fuse must be one of {'LL','concat','add'}")
        else:
            first_in = input_channels
            self.reduce = None

        # layers = []
        # in_channels = first_in
        # for out_channels in conv_channels:
        #     layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        #     layers.append(nn.SiLU(inplace=True))
        #     in_channels = out_channels

        # self.conv_features = nn.Sequential(*layers)
        blocks = []
        in_channels = first_in
        for out_channels in conv_channels:
            blocks.append(ResidualConvBlock(in_channels, out_channels, k=3, s=1, p=1))
            in_channels = out_channels
        self.conv_features = nn.Sequential(*blocks)
        self.final_channels = conv_channels[-1]  # 输出通道

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # DWT 下采样（H,W -> H/2,W/2）
        if self.use_dwt:
            LL, LH, HL, HH = self.dwt(x)
            if self.dwt_fuse == 'LL':
                z = LL
            elif self.dwt_fuse == 'concat':
                z = torch.cat([LL, LH, HL, HH], dim=1)
            else:  # 'add'
                z = LL + self.reduce(torch.cat([LL, LH, HL, HH], dim=1))
        else:
            z = x
        # 卷积特征
        return self.conv_features(z)


class VisionLSTM2(nn.Module):
    def __init__(self, dim, input_shape, patch_size, depth, output_shape, mode, pooling,
                 drop_path_rate, drop_path_decay, stride, legacy_norm, conv_kind,
                 conv_kernel_size, proj_bias, norm_bias, feature_extractor_channels, use_fourier=False):
        super(VisionLSTM2, self).__init__()

        self.dim = dim
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
        
        # 1) 用 DWT 替代 FFT（仍然沿用 use_fourier 变量作为开关）
        self.feature_extractor = FeatureExtractor(
            input_channels=input_shape[0],
            conv_channels=feature_extractor_channels,
            use_fourier=use_fourier,    # 现在等价于 use_dwt
            use_dwt=use_fourier,        # 显式传入
            #dwt_fuse='LL'               # 'LL'（推荐）/'concat'/'add'
            dwt_fuse='concat'
        )
        
        # 2) DWT 不再增加通道，PatchEmbed 的输入通道就是卷积输出通道
        num_channels = feature_extractor_channels[-1]

        # 3) 若启用 DWT，下采样一半；确保 patch_size 能整除新的 H,W
        if use_fourier:  # DWT on
            pe_res = (input_shape[1] // 2, input_shape[2] // 2)
        else:
            pe_res = (input_shape[1], input_shape[2])
        
        self.patch_embed = VitPatchEmbed(
            dim=dim,
            num_channels=num_channels,
            resolution=pe_res,
            patch_size=patch_size,
            stride=patch_size,
            init_weights="xavier_uniform"
        )

        # Initialize learnable positional embedding
        self.pos_embed = VitPosEmbed2d(seqlens=self.patch_embed.seqlens, dim=dim)

        # Initialize blocks
        if drop_path_decay and drop_path_rate > 0.:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        else:
            dpr = [drop_path_rate] * depth

        self.blocks = nn.ModuleList(
            [
                ViLBlockPair(
                    dim=dim,
                    drop_path=dpr[i],
                    conv_kind=conv_kind,
                    seqlens=self.patch_embed.seqlens,
                    proj_bias=proj_bias,
                    norm_bias=norm_bias,
                )
                for i in range(depth)
            ],
        )
        if pooling == "bilateral_flatten":
            head_dim = dim * 2
        else:
            head_dim = dim
        self.norm = LayerNorm(dim, bias=norm_bias, eps=1e-6)
        if legacy_norm:
            self.legacy_norm = nn.LayerNorm(head_dim)
        else:
            self.legacy_norm = nn.Identity()

        # Classification head
        if mode == "features":
            assert self.output_shape is None
            self.head = None
            if self.pooling is None:
                self.output_shape = (self.patch_embed.num_patches, dim)
            elif self.pooling == "to_image":
                self.output_shape = (dim, *self.patch_embed.seqlens)
            else:
                raise NotImplementedError(f"invalid pooling '{pooling}' for mode '{mode}'")
        elif mode == "classifier":
            assert self.output_shape is not None and len(self.output_shape) == 1, \
                f"define number of classes via output_shape=(num_classes,) (e.g. output_shape=(1000,) for ImageNet-1K"
            #self.head = nn.Linear(dim * 3, self.output_shape[0])  # Updated to match combined feature size
            #nn.init.trunc_normal_(self.head.weight, std=2e-5)
            #nn.init.zeros_(self.head.bias)
            self.head = nn.Linear(head_dim, self.output_shape[0])
            nn.init.trunc_normal_(self.head.weight, std=2e-5)
            nn.init.zeros_(self.head.bias)
            self.head_adapter = HeadResidualAdapter(head_dim=head_dim, branch_dim=dim)
        else:
            raise NotImplementedError

        # Branch for additional feature maps with convolutional layer
        # Adjust `in_channels` to match the output channels of `feature_extractor`
        branch_in = feature_extractor_channels[-1]
        self.feature_extractor_branch = nn.Sequential(
            nn.Conv2d(in_channels=branch_in, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(8),     # 固定到 8x8，避免分辨率依赖
            nn.Flatten(),
            #nn.Linear(32 * 8 * 8, 256),
            nn.Linear(32 * 8 * 8, 384),
            nn.SiLU(),
            #nn.Linear(256, dim)
            nn.Linear(384, dim)
        )
    
        g_h, g_w = self.patch_embed.seqlens   # e.g., 8x8 或 4x4
        assert g_h == g_w
        self.mixers = nn.ModuleList([ResidualDepthwiseMix(self.dim, grid=g_h) for _ in range(self.depth)])

    def load_state_dict(self, state_dict, strict=True):
        old_pos_embed = state_dict["pos_embed.embed"]
        if old_pos_embed.shape != self.pos_embed.embed.shape:
            state_dict["pos_embed.embed"] = interpolate_sincos(embed=old_pos_embed, seqlens=self.pos_embed.seqlens)
        return super().load_state_dict(state_dict=state_dict, strict=strict)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed.embed"}

    def forward(self, x):
        # Main branch
        x_main = self.feature_extractor(x)
        x_main = self.patch_embed(x_main)
        x_main = self.pos_embed(x_main)
        x_main = einops.rearrange(x_main, "b ... d -> b (...) d")
        
        for i, block in enumerate(self.blocks):
            x_main = block(x_main)
            if (i % 2) == 0:   # 每隔一层插一次，控制开销
                x_main = self.mixers[i](x_main)
        x_main = self.norm(x_main)
    
        if self.pooling is None:
            x_main = self.legacy_norm(x_main)
        elif self.pooling == "to_image":
            x_main = self.legacy_norm(x_main)
            seqlen_h, seqlen_w = self.patch_embed.seqlens
            x_main = einops.rearrange(
                x_main,
                "b (seqlen_h seqlen_w) dim -> b dim seqlen_h seqlen_w",
                seqlen_h=seqlen_h,
                seqlen_w=seqlen_w,
            )
        elif self.pooling == "bilateral_avg":
            x_main = (x_main[:, 0] + x_main[:, -1]) / 2
            x_main = self.legacy_norm(x_main)
        elif self.pooling == "bilateral_flatten":
            x_main = torch.concat([x_main[:, 0], x_main[:, -1]], dim=1)
            x_main = self.legacy_norm(x_main)
        else:
            raise NotImplementedError(f"pooling '{self.pooling}' is not implemented")
    
        # Feature extractor branch
        #feature_maps = self.feature_extractor(x)
        #feature_branch_out = self.feature_extractor_branch(feature_maps)
        feature_maps = self.feature_extractor(x)
        feature_branch_out = self.feature_extractor_branch(feature_maps)   # (B, dim)
    
        # Combine both branches
        #combined_features = torch.cat([x_main, feature_branch_out], dim=1)
        x_main = self.head_adapter(x_main, feature_branch_out)             # (B, head_dim)
        combined_output = self.head(x_main)
        return combined_output
        # Final classification head
        # if self.head is not None:
        #     combined_output = self.head(combined_features)
        # else:
        #     combined_output = combined_features

        # return combined_output

