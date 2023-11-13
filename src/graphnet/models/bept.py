"""BERT Pre-training of Pulse Transformers."""


from icecream import ic
import torch
from torch import nn
from torch_geometric.data import Data
from timm.models.layers import drop_path, trunc_normal_
import math
from typing import Optional, List, Callable, Tuple, Union, Set, Any


class MLP(nn.Module):
    """Multilayer Perceptron."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Optional[Callable] = None,
        drop: Optional[float] = 0.0,
    ):
        """Initialize the MLP.

        Args:
            in_features: Number of input features.
            hidden_features: Number of hidden features. Defaults to in_features.
            out_features: Number of output features. Defaults to in_features.
            act_layer: Activation layer.
            drop: Dropout probability.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.act_layer = act_layer or nn.GELU()
        self.drop = nn.Dropout(drop)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of MLP.

        Args:
            x: Input data.

        Returns: Output of MLP.
        """
        x = self.fc1(x)
        x = self.act_layer(x)
        x = self.drop(x)  # Can be removed.
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RelativeAttention(nn.Module):
    """Relative attention module."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_head_dim: Optional[int] = None,
    ):
        """Relative attention module.

        Args:
            dim: Number of features.
            num_heads: Number of attention heads.
            qkv_bias: If true include bias in query, key, and value projections.
            qk_scale: Not sure.
            attn_drop: Dropout probability of attention.
            proj_drop: Dropout probability of projection.
            attn_head_dim: Number of features per attention head.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.proj_q = nn.Linear(dim, all_head_dim, bias=qkv_bias)
        self.proj_k = nn.Linear(dim, all_head_dim, bias=qkv_bias)
        self.proj_v = nn.Linear(dim, all_head_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        rel_pos_bias: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the RelativeAttention.

        Args:
            q: Query.
            k: Key.
            v: Value.
            rel_pos_bias: Relative position bias.
            key_padding_mask: Padding mask of the input data.

        Returns: Output of the RelativeAttention.
        """
        B, N, C = q.shape
        q = (
            self.proj_q(q)
            .reshape(B, N, self.num_heads, -1)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.proj_k(k)
            .reshape(B, N, self.num_heads, -1)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.proj_v(v)
            .reshape(B, N, self.num_heads, -1)
            .permute(0, 2, 1, 3)
        )
        q *= self.scale
        attn = q @ k.transpose(-2, -1)
        if rel_pos_bias is not None:
            bias = torch.einsum("bhic,bijc->bhij", q, rel_pos_bias)
            attn = attn + bias

        if key_padding_mask is not None:
            assert (
                key_padding_mask.dtype == torch.float32
                or key_padding_mask.dtype == torch.float16
            ), "Only float mask is supported."
            bias = torch.min(
                key_padding_mask[:, None, :], key_padding_mask[:, :, None]
            )
            bias[
                torch.max(
                    key_padding_mask[:, None, :], key_padding_mask[:, :, None]
                )
                < 0
            ] = 0
            attn = attn + bias.unsqueeze(1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2)
        if rel_pos_bias is not None:
            x += torch.einsum("bhij,bijc->bihc", attn, rel_pos_bias)
        x = x.reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    """Standard transformer block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,  # noqa: F811
        init_values: Optional[float] = None,
        act_layer: Optional[Callable] = None,
        norm_layers: Optional[List[Callable]] = None,
    ):
        """Initialize the TransformerBlock.

        Args:
            dim: Number of features.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio between hidden and primary dimension.
            drop: Dropout probability of feedforward.
            attn_drop: Dropout probability of attention.
            drop_path: Not used. (Path dropout probability, see: https://arxiv.org/abs/1605.07648)
            init_values: Initial value for the gamma parameters.
            act_layer: Activation layer.
            norm_layers: Normalization layers.
        """
        super().__init__()
        if norm_layers is None:
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(mlp_ratio * dim)
        self.attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=attn_drop,
            batch_first=True,
        )
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if init_values is not None:
            self.gamma1 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
            self.gamma2 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
        else:
            self.gamma1 = self.gamma2 = None

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the TransformerBlock.

        Args:
            x: Input data.
            attn_mask: Attention mask.
            key_padding_mask: Padding mask of input data.

        Returns: Output data.
        """
        xn = self.norm1(x)
        if self.gamma1 is None:
            x = (
                x
                + self.attn(
                    xn,
                    xn,
                    xn,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                )[0]
            )
            x = x + self.mlp(self.norm2(x))
        else:
            x = (
                x
                + self.gamma1
                * self.attn(
                    xn,
                    xn,
                    xn,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                )[0]
            )
            x = x + self.gamma2 * self.mlp(self.norm2(x))
        return x


class RelAttnTransformerBlock(nn.Module):
    """Transformer block with relative attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,  # noqa: F811
        init_values: Optional[float] = None,
        act_layer: Optional[Callable] = None,
        norm_layers: Optional[List[Callable]] = None,
    ):
        """Initialize the RelAttnTransformerBlock.

        Args:
            dim: Primary dimension of the model.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of hidden to primary dimension.
            qkv_bias: If True, add bias to the query, key, and value projections.
            qk_scale: Don't know.
            drop: Value for dropout in feedforward.
            attn_drop: Value for dropout in attention.
            drop_path: Not used. (Path dropout probability, see: https://arxiv.org/abs/1605.07648)
            init_values: Initial value for the gamma parameters.
            act_layer: Activation layer.
            norm_layers: Normalization layers.
        """
        super().__init__()
        if norm_layers is None:
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)

        self.attn = RelativeAttention(
            dim,
            num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
        )

        mlp_hidden_dim = int(mlp_ratio * dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if init_values is not None:
            self.gamma1 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
            self.gamma2 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
        else:
            self.gamma1 = self.gamma2 = None

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rel_pos_bias: Optional[torch.Tensor] = None,
        kv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the RelAttnTransformerBlock.

        Args:
            x: Input data.
            key_padding_mask: Mask of the input data.
            rel_pos_bias: Relative position attention bias.
            kv: Key and Value for the MultiheadAttention.

        Returns: Output of the RelAttnTransformerBlock.
        """
        xn = self.norm1(x)
        kv = xn if kv is None else self.norm1(kv)
        if self.gamma1 is None:
            x = x + self.attn(xn, kv, kv, rel_pos_bias, key_padding_mask)
            x = x + self.mlp(self.norm2(x))
        else:
            x = x + self.gamma1 * self.attn(
                xn, kv, kv, rel_pos_bias, key_padding_mask
            )
            x = x + self.gamma2 * self.mlp(self.norm2(x))
        return x


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding."""

    def __init__(self, dim: int, M: int = 10_000):
        """Initialize the SinusoidalPosEmb.

        Args:
            dim: Dimension of the embedding.
            M: Scalar for the Sinusoidal positional embedding.
        """
        super().__init__()
        self.dim = dim
        self.M = M

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input data.

        Returns: (Positional) Fourier embedded input data.
        """
        device = x.device
        half_dim = self.dim // 2
        emb: torch.Tensor
        emb = math.log(self.M) / half_dim  # type: ignore
        emb = torch.exp(torch.arange(half_dim, device=device) * (-emb))
        emb = x[Ellipsis, None] * emb[None, Ellipsis]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class Extractor(nn.Module):
    """Extractor module."""

    _pos_idx: List[int] = [0, 1, 2]
    _t_idx: List[int] = [3]
    _remaining_idx: List[int] = [4, 5, 6]

    def __init__(self, dim_base: int = 128, dim: int = 384) -> None:
        """Initialize the Extractor module.

        Args:
            dim_base: Base dimension of the model.
            dim: Core dimension of the model.
        """
        super().__init__()
        self.emb = SinusoidalPosEmb(dim=dim_base)
        self.proj = nn.Sequential(
            nn.Linear(
                (
                    n_feat := (
                        len(self._pos_idx)
                        + len(self._t_idx)
                        + len(self._remaining_idx)
                    )
                )
                * dim_base,
                n_feat * dim_base,
            ),
            nn.LayerNorm(n_feat * dim_base),
            nn.GELU(),
            nn.Linear(n_feat * dim_base, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Scaled input data.

        Returns: Output of the extractor.
        """
        pos = x[Ellipsis, self._pos_idx]
        t = x[Ellipsis, self._t_idx]
        remaining = x[Ellipsis, self._remaining_idx]
        x = torch.cat(
            [
                self.emb(4096 * pos).flatten(-2),
                self.emb(4096 * t).flatten(-2),
                self.emb(1024 * remaining).flatten(-2),
            ],
            dim=-1,
        )
        x = self.proj(x)
        return x


class RelativeSpaceTime(nn.Module):
    """RelativeSpaceTime attention-bias."""

    _pos_idx: List[int] = [0, 1, 2]
    _t_idx: List[int] = [3]
    _count = 2 * (len(_pos_idx) + len(_t_idx))
    _time_scale = 3e4 / 500 * 3e-1  # Detector dependent.

    def __init__(self, dim: int):
        """Initialize the RelativeSpaceTime attention-bias module.

        Args:
            dim: Dimension of the model.
        """
        super().__init__()
        self.emb = SinusoidalPosEmb(dim=dim)
        self.proj1d = nn.Linear(self._count, 1)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the RelativeSpaceTime attention-bias module.

        Args:
            x: Input data, containing position and time values.

        Returns: Relative attention and embedding.
        """
        pos = x[Ellipsis, self._pos_idx]
        t = x[Ellipsis, self._t_idx]
        delta_pos = pos[:, :, None] - pos[:, None, :]
        delta_t = t[:, :, None] - t[:, None, :] * self._time_scale
        delta_pos_sq = delta_pos.pow(2)
        delta_t_sq = delta_t.pow(2)

        ds2 = delta_pos_sq.sum(dim=-1) - delta_t_sq.squeeze(-1)
        d = torch.sign(ds2) * torch.sqrt(torch.abs(ds2))
        x = torch.cat(
            [
                delta_pos,
                delta_t,
                delta_pos_sq,
                delta_t_sq,
            ],
            dim=-1,
        )
        x = self.proj1d(x).squeeze(-1)
        d += x
        emb = self.emb(1024 * d.clip(-4, 4))
        rel_attn = self.proj(emb)
        return rel_attn, emb


class LocalBlock(nn.Module):
    """LocalBlock, not sure if this is used."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        init_values: float = 1.0,
    ):
        """Initialize Local attention transformer block.

        Args:
            dim: Number of features of the transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio between hidden layer and latent later.
            init_values: Initial value for the gamma parameters.
        """
        super().__init__()
        self.proj_rel_bias = nn.Linear(dim // num_heads, dim // num_heads)
        self.block = RelAttnTransformerBlock(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            init_values=init_values,
        )

    def forward(
        self,
        x: torch.Tensor,
        nbs: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rel_pos_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the LocalBlock.

        Args:
            x: Input data.
            nbs: Neighbors of the input data.
            key_padding_mask: Mask of the input data.
            rel_pos_bias: Relative position bias.

        Returns: Output of the LocalBlock.
        """
        B, max_length, C = x.shape
        mask = (
            key_padding_mask
            if key_padding_mask is not None
            else torch.ones(B, max_length, dtype=torch.bool, device=x.device)
        )
        m = torch.gather(mask.unsqueeze(1).expand(-1, max_length, -1), 2, nbs)
        attn_mask = torch.zeros(m.shape, device=m.device)
        attn_mask[~mask] = -torch.inf
        attn_mask = attn_mask[mask]

        if rel_pos_bias is not None:
            rel_pos_bias = torch.gather(
                rel_pos_bias,
                2,
                nbs.unsqueeze(-1).expand(-1, -1, -1, rel_pos_bias.shape[-1]),
            )
            rel_pos_bias = rel_pos_bias[mask]
            rel_pos_bias = self.proj_rel_bias(rel_pos_bias).unsqueeze(1)

        x1 = torch.gather(
            x.unsqueeze(1).expand(-1, max_length, -1, -1),
            2,
            nbs.unsqueeze(-1).expand(-1, -1, -1, C),
        )
        x1 = x1[mask]
        x1 = self.block(
            x1[:, :1],
            rel_pos_bias=rel_pos_bias,
            key_padding_mask=attn_mask[:, :1],
            kv=x1,
        )
        x = torch.zeros(x.shape, device=x.device, dtype=x1.dtype)
        x[mask] = x1.squeeze(1)
        return x


class DeepIceModelNoExtractor(nn.Module):
    """Second place Kaggle model Transformer only."""

    def __init__(
        self,
        dim: int = 384,
        depth: int = 12,
        head_size: int = 32,
        depth_rel: int = 4,
        n_rel: int = 1,
    ) -> None:
        """Instantiate DeepIceModelNoExtractor.

        Args:
            dim: Number of features.
            depth: Depth of the model.
            head_size: Number of features per attention head.
            depth_rel: Number of relative attention layers.
            n_rel: Layer at which to calculate relative attention.
        """
        super().__init__()
        self.rel_pos = RelativeSpaceTime(head_size)
        self.sandwich = nn.ModuleList(
            [
                RelAttnTransformerBlock(dim=dim, num_heads=dim // head_size)
                for i in range(depth_rel)
            ]
        )
        self.cls_token = nn.Linear(dim, 1, bias=False)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    num_heads=dim // head_size,
                    mlp_ratio=4,
                    drop_path=0.0 * (i / (depth - 1)),
                    init_values=1,
                )
                for i in range(depth)
            ]
        )
        self.apply(self._init_weights)
        trunc_normal_(self.cls_token.weight, std=0.02)
        self.n_rel = n_rel

    def fix_init_weight(self) -> None:
        """No clue what the purpose of this is.

        Returns: None
        """

        def rescale(param: Any, layer_id: Any) -> None:
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self) -> None:
        """Initialize the weights of the model.

        Returns: None
        """

        def _init_weights(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set[Any]:
        """Stuff for jit.

        Returns: Set of parameters that should not be weight decayed.
        """
        return {"cls_token"}

    def forward(
        self, x: torch.Tensor, x_orig: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: The input (extracted) data.
            x_orig: Original data used to create SpaceTime-attention bias.
            mask: Mask of the input data.

        Returns:
            The output of the model.
        """
        Lmax = mask.sum(-1).max()
        rel_pos_bias, rel_enc = self.rel_pos(x_orig, Lmax)
        # nbs = get_nbs(x0, Lmax)
        mask = mask[:, :Lmax]
        B, _ = mask.shape
        attn_mask = torch.zeros(mask.shape, device=mask.device)
        attn_mask[~mask] = -torch.inf

        for i, blk in enumerate(self.sandwich):
            x = blk(x, attn_mask, rel_pos_bias)
            if i + 1 == self.n_rel:
                rel_pos_bias = None

        mask = torch.cat(
            [torch.ones(B, 1, dtype=mask.dtype, device=mask.device), mask], 1
        )
        attn_mask = torch.zeros(mask.shape, device=mask.device)
        attn_mask[~mask] = -torch.inf
        cls_token = self.cls_token.weight.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([cls_token, x], 1)

        for blk in self.blocks:
            x = blk(x, None, attn_mask)
        return x
