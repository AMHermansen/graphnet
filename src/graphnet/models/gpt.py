"""
Andrey Karpathy NanoGPT source code. Modified to work on pulsemaps
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""
import dataclasses
from typing import Optional, Dict, Tuple, Any

import numpy as np
from attr import define
from icecream import ic
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch

from graphnet.models.bept import Extractor, SimpleExtractor
from graphnet.models.detector import Detector
from graphnet.models.model import Model

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class LayerNorm(Model):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(Model):

    def __init__(
            self,
            n_head: int,
            n_embd: int,
            ln_bias: int,
            dropout: float,
            block_size: int = 256,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=ln_bias)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=ln_bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.warning("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                        .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of scaled dot-product attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(Model):

    def __init__(
            self,
            n_embd: int,
            mlp_bias: bool,
            dropout: float,
    ):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=mlp_bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=mlp_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


# # config for nano gpt
# @dataclass
# class GPTConfig:
#     block_size: int = 1024
#     vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
#     n_layer: int = 12
#     n_head: int = 12
#     n_embd: int = 768
#     dropout: float = 0.0
#     bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class GPTBlock(Model):

    def __init__(
            self,
            n_head: int,
            n_embd: int = 768,
            ln_bias: bool = True,
            mlp_bias: bool = True,  # bias should probably be set to false.
            block_size: int = 256,
            dropout: float = 0.0,
    ):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=ln_bias)
        self.attn = CausalSelfAttention(
            n_head,
            n_embd,
            ln_bias,
            dropout,
            block_size,
        )
        self.ln_2 = LayerNorm(n_embd, bias=ln_bias)
        self.mlp = MLP(
            n_embd,
            mlp_bias,
            dropout,
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    n_sensors: int = 5504
    block_size: int = 256
    n_embd: int = 192
    n_head: int = 12
    ln_bias: bool = False
    mlp_bias: bool = False
    n_layer: int = 12
    dropout: float = 0.0


class GPT(Model):

    def __init__(
            self,
            n_sensors: int = 5504,
            block_size: int = 256,
            n_embd: int = 192,
            n_head: int = 12,
            ln_bias: bool = False,
            mlp_bias: bool = False,
            n_layer: int = 12,
            dropout: float = 0.0,
    ):
        super().__init__(
        )

        assert block_size is not None
        self._n_sensors = n_sensors
        self._block_size = block_size
        self._n_embd = n_embd
        self._ln_bias = ln_bias
        self._mlp_bias = mlp_bias

        self.transformer = nn.ModuleDict(dict(
            extractor=SimpleExtractor(128, n_embd),
            wpe=nn.Embedding(block_size, n_embd),
            drop=nn.Dropout(dropout),
            h=nn.ModuleList([GPTBlock(
                n_head=n_head,
                n_embd=n_embd,
                ln_bias=ln_bias,
                mlp_bias=mlp_bias,
                block_size=block_size,
                dropout=dropout,
            ) for _ in range(n_layer)]),
            ln_f=LayerNorm(n_embd, bias=ln_bias),
        ))
        self.lm_head = nn.Linear(n_embd, n_sensors, bias=False)
        self.time_head = nn.Linear(n_embd, 1, bias=True)
        self.charge_head = nn.Linear(n_embd, 1, bias=True)

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, data: Data, targets=None):
        x, mask = to_dense_batch(data.x, data.batch)

        device = x.device
        b, t, c = x.size()
        assert t <= self._block_size, f"Cannot forward sequence of length {t}, block size is only {self._block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.extractor(x)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)[mask]
            time_pred = self.time_head(x)[mask]
            charge_pred = self.charge_head(x)[mask]
            loss_ce = F.cross_entropy(logits, targets)
            loss_t = F.mse_loss(time_pred, data.t_shifted)
            loss_q = F.mse_loss(charge_pred, data.q_shifted)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            time_pred = self.time_head(x[:, [-1], :])
            charge_pred = self.charge_head(x[:, [-1], :])
            loss_ce = None
            loss_t = None
            loss_q = None
        return {
            "pred_logits": logits,
            "pred_time": time_pred,
            "pred_charge": charge_pred,
            "loss_ce": loss_ce,
            "loss_t": loss_t,
            "loss_q": loss_q
        }

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    def id_to_vars(self, id, detector: Detector):
        id = id.detach().cpu().numpy().flatten()
        mask = id < len(detector.geometry_table)
        id = id[mask]
        data_matrix = detector.geometry_table.iloc[id][["dom_x", "dom_y", "dom_z", "rde", "pmt_area"]].values
        x = detector.feature_map()["dom_x"](data_matrix[:, 0])
        y = detector.feature_map()["dom_y"](data_matrix[:, 1])
        z = detector.feature_map()["dom_y"](data_matrix[:, 2])
        rde = detector.feature_map()["rde"](data_matrix[:, 3])
        pmt_area = detector.feature_map()["pmt_area"](data_matrix[:, 4])
        processed_values = np.stack([x, y, z, rde, pmt_area], axis=1)
        return torch.from_numpy(processed_values), torch.from_numpy(mask).unsqueeze(1)

    @torch.no_grad()
    def generate(self, x: Data, max_new_tokens, detector, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        dtype = x.x.dtype
        device = x.x.device
        finished = {
            k: None for k in set(x.batch)
        }
        finished_runs = torch.zeros(len(finished))
        for _ in range(max_new_tokens):
            # idx = self.transformer.extractor(x)  # Extract raw-pulsemap to latentspace
            # # if the sequence context is growing too long we must crop it at block_size
            # idx_cond = idx if idx.size(1) <= self._block_size else idx[:, -self._block_size:]
            # forward the model to get the logits for the index in the sequence
            out_dict = self(x)
            logits = out_dict["pred_logits"]
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature  # Boltzmann-distribution
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            all_detector_var, mask_next = self.id_to_vars(idx_next, detector)
            new_finished_runs = torch.logical_or(finished_runs, ~mask_next)
            just_finished_runs = torch.logical_and(new_finished_runs, ~finished_runs)
            just_finished_runs_idx = torch.arange(len(finished))[just_finished_runs]

            # append sampled index to the running sequence and continue
            x_val = all_detector_var[:, 0].unsqueeze(1).unsqueeze(1)
            y_val = all_detector_var[:, 1].unsqueeze(1).unsqueeze(1)
            z_val = all_detector_var[:, 2].unsqueeze(1).unsqueeze(1)
            rde = all_detector_var[:, 3].unsqueeze(1).unsqueeze(1)
            pmt_area = all_detector_var[:, 4].unsqueeze(1).unsqueeze(1)
            predicted_next = torch.cat(
                (
                    x_val,
                    y_val,
                    z_val,
                    out_dict["pred_time"],
                    out_dict["pred_charge"],
                    rde,
                    pmt_area,
                ),
                dim=2
            )
            block_dense, mask_dense = to_dense_batch(x.x, x.batch)
            block_dense_full = torch.cat((block_dense, predicted_next), dim=1)
            mask_dense_full = torch.cat((mask_dense, mask_next), dim=1)
            B, L, _ = block_dense_full.shape
            batch = torch.stack([torch.arange(B) for _ in range(L)], dim=1)
            x = Data(x=block_dense_full[mask_dense_full].to(dtype), batch=batch[mask_dense_full])
        return x


class GPTLit(Model):
    def __init__(
            self,
            gpt_config: GPTConfig = None,
            optimizer_class: type = torch.optim.Adam,
            optimizer_kwargs: Optional[Dict] = None,
            scheduler_class: Optional[type] = None,
            scheduler_kwargs: Optional[Dict] = None,
            scheduler_config: Optional[Dict] = None,
            loss_weights: Tuple[float, float, float] = (1., 1., 1.)
    ):
        super().__init__(name=__name__, class_name=self.__class__.__name__)
        self._gpt_config = gpt_config or GPTConfig()
        self._optimizer_class = optimizer_class
        self._optimizer_kwargs = optimizer_kwargs or {}
        self._scheduler_class = scheduler_class
        self._scheduler_kwargs = scheduler_kwargs or {}
        self._scheduler_config = scheduler_config or {}
        self.loss_weights = loss_weights

        self.gpt = GPT(**dataclasses.asdict(self._gpt_config))

        # self.save_hyperparameters()

    def _shift_sensors(self, data, num_classes=5484):
        """Shift sensor ids to be for next time step."""
        sensor_id, mask = to_dense_batch(data.sensor_id, data.batch, fill_value=num_classes)
        B, L = sensor_id.shape
        sensor_id_shifted = torch.cat(
            [
                sensor_id,
                torch.ones(B, 1, dtype=torch.long, device=sensor_id.device)
                * num_classes
            ],
            dim=1
        )
        sensor_id_shifted = sensor_id_shifted[:, 1:]
        data.sensor_id_shifted = sensor_id_shifted[mask].to(torch.long)
        return data

    def _add_shifted_tq(self, data):
        x, mask = to_dense_batch(data.x, data.batch)
        t = x[..., [3]]
        q = x[..., [4]]
        B, *_ = t.shape

        t = torch.cat(
            [
                t,
                torch.zeros(B, 1, 1, dtype=t.dtype, device=t.device)
            ],
            dim=1
        )[:, 1:]
        q = torch.cat(
            [
                q,
                torch.zeros(B, 1, 1, dtype=t.dtype, device=q.device)
            ],
            dim=1
        )[:, 1:]

        data.t_shifted = t[mask]
        data.q_shifted = q[mask]
        return data

    def _shared_step(self, batch: Data, batch_idx: int):
        data = self._shift_sensors(batch)
        data = self._add_shifted_tq(data)
        out_dict = self.gpt(data, data.sensor_id_shifted)
        return out_dict

    def compute_loss(self, out_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        loss_ce = out_dict["loss_ce"]
        loss_t = out_dict["loss_t"]
        loss_q = out_dict["loss_q"]
        loss = (
                self.loss_weights[0] * loss_ce
                + self.loss_weights[1] * loss_t
                + self.loss_weights[2] * loss_q
        )
        return loss

    def training_step(self, batch: Data, batch_idx: int) -> Dict[str, torch.Tensor]:
        out_dict = self._shared_step(batch, batch_idx)
        loss = self.compute_loss(out_dict)
        self.log("train_loss", loss, prog_bar=True, logger=True,
                 batch_size=self._get_batch_size(batch))
        self.log("train_loss_ce", self.loss_weights[0] * out_dict["loss_ce"], prog_bar=True, logger=True,
                 batch_size=self._get_batch_size(batch))
        self.log("train_loss_t", self.loss_weights[1] * out_dict["loss_t"], prog_bar=True, logger=True,
                 batch_size=self._get_batch_size(batch))
        self.log("train_loss_q", self.loss_weights[2] * out_dict["loss_q"], prog_bar=True, logger=True,
                 batch_size=self._get_batch_size(batch))
        return {"loss": loss, **out_dict}

    def validation_step(self, batch: Data, batch_idx: int) -> Dict[str, torch.Tensor]:
        out_dict = self._shared_step(batch, batch_idx)
        loss = self.compute_loss(out_dict)
        self.log("val_loss", loss, prog_bar=True, logger=True,
                 batch_size=self._get_batch_size(batch))
        self.log("val_loss_ce", self.loss_weights[0] * out_dict["loss_ce"], prog_bar=True, logger=True,
                 batch_size=self._get_batch_size(batch))
        self.log("val_loss_t", self.loss_weights[1] * out_dict["loss_t"], prog_bar=True, logger=True,
                 batch_size=self._get_batch_size(batch))
        self.log("val_loss_q", self.loss_weights[2] * out_dict["loss_q"], prog_bar=True, logger=True,
                 batch_size=self._get_batch_size(batch))
        return {"loss": loss, **out_dict}

    def predict_step(self, batch: Data, batch_idx: int) -> Dict[str, torch.Tensor]:
        out_dict = self._shared_step(batch, batch_idx)
        data = self._shift_sensors(batch)
        data = self._add_shifted_tq(data)
        return {"target_t": data.t_shifted, "target_q": data.q_shifted, "target_sensor": data.sensor_id_shifted, **out_dict}

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure the model's optimizer(s)."""
        optimizer = self._optimizer_class(
            self.parameters(), **self._optimizer_kwargs
        )
        config = {
            "optimizer": optimizer,
        }
        if self._scheduler_class is not None:
            scheduler = self._scheduler_class(
                optimizer, **self._scheduler_kwargs
            )
            config.update(
                {
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        **self._scheduler_config,
                    },
                }
            )
        return config


if __name__ == "__main__":
    gpt = GPT()

    x = torch.rand(10, 7)
    ba = torch.tensor([0, 0, 1, 1, 1, 1, 1, 2, 2, 2])
    sensors = torch.from_numpy(np.random.choice(np.arange(5504, dtype=np.longlong), size=(10)))
    # sensors, _ = to_dense_batch(sensors, ba)
    dat = Data(x=x, batch=ba)

    from graphnet.models.detector import IceCube86
    detector = IceCube86()
    generated = gpt.generate(dat, max_new_tokens=10, detector=detector)
    print(generated)
