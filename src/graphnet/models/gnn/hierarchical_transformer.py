from typing import Union

from graphnet.models.gnn import GNN

import torch
from torch.nn import (
    TransformerEncoderLayer,
    TransformerEncoder,
    Linear,
    ModuleList,
    GELU,
    LayerNorm,
    Module,
    Parameter
)
from torch_geometric.utils import to_dense_batch


class MLP(Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.linear1 = Linear(input_dim, hidden_dim)
        self.linear2 = Linear(hidden_dim, output_dim)
        self.act = GELU()
        self.norm = LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.linear2(x)
        return x


class HierarchicalTransformer(GNN):
    def __init__(
            self,
            hierarchical_dimensions: Union[int, list[int]],
            hierarchical_heads: Union[int, list[int]],
            hierarchical_layers: Union[int, list[int]],
            input_mlp_hidden: int,
            input_dim: int,
            upscale_mlp_hidden: int,
    ):
        super().__init__(nb_inputs=0, nb_outputs=0)
        self.hierarchical_dimensions = hierarchical_dimensions if isinstance(hierarchical_dimensions, list) else [hierarchical_dimensions for _ in range(3)]
        self.hierarchical_heads = hierarchical_heads if isinstance(hierarchical_heads, list) else [hierarchical_heads for _ in range(3)]
        self.hierarchical_layers = hierarchical_layers if isinstance(hierarchical_layers, list) else [hierarchical_layers for _ in range(3)]
        self.input_mlp_hidden = input_mlp_hidden
        self.upscale_mlp_hidden = upscale_mlp_hidden

        self._nb_inputs = input_dim
        self._nb_outputs = hierarchical_dimensions[-1]

        self.cls_stage1 = Parameter(torch.randn(1, 1, hierarchical_dimensions[0]))
        self.cls_stage2 = Parameter(torch.randn(1, 1, hierarchical_dimensions[1]))
        self.cls_stage3 = Parameter(torch.randn(1, 1, hierarchical_dimensions[2]))

        self.input_mlp = MLP(
            input_dim=input_dim,
            hidden_dim=input_mlp_hidden,
            output_dim=hierarchical_dimensions[0],
        )
        self.stage12_mlp = MLP(
            input_dim=hierarchical_dimensions[0],
            hidden_dim=int(self.upscale_mlp_hidden*self.hierarchical_dimensions[0]),
            output_dim=hierarchical_dimensions[1],
        )
        self.stage23_mlp = MLP(
            input_dim=hierarchical_dimensions[1],
            hidden_dim=int(self.upscale_mlp_hidden*self.hierarchical_dimensions[1]),
            output_dim=hierarchical_dimensions[2],
        )

        self.stage1_transformer = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=hierarchical_dimensions[0],
                nhead=hierarchical_heads[0],
                batch_first=True,
            ),
            num_layers=hierarchical_layers[0],
        )
        self.stage2_transformer = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=hierarchical_dimensions[1],
                nhead=hierarchical_heads[1],
                batch_first=True,
            ),
            num_layers=hierarchical_layers[1],
        )
        self.stage3_transformer = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=hierarchical_dimensions[2],
                nhead=hierarchical_heads[2],
                batch_first=True,
            ),
            num_layers=hierarchical_layers[2],
        )

    def forward(self, data):
        x, _ = to_dense_batch(data.x, data.batch)
        batch, pulses, strings, doms, features = x.size()
        x = x.permute(0, 2, 3, 1, 4).reshape(batch * strings * doms, pulses, features)
        x = self.input_mlp(x)
        x = torch.cat((self.cls_stage1.repeat(x.size(0), 1, 1), x), dim=-2)
        x = self.stage1_transformer(x)
        x = x[:, 0, :]
        x = x.view(batch*strings, doms, -1)
        x = self.stage12_mlp(x)
        x = torch.cat((self.cls_stage2.repeat(x.size(0), 1, 1), x), dim=-2)
        x = self.stage2_transformer(x)
        x = x[:, 0, :]
        x = x.view(batch, strings, -1)
        x = self.stage23_mlp(x)
        x = torch.cat((self.cls_stage3.repeat(x.size(0), 1, 1), x), dim=-2)
        x = self.stage3_transformer(x)
        x = x[:, 0, :]
        return x
