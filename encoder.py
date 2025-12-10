"""
Graph Attention Encoder Module
Extracted encoder layers from the Attention-based TSP solver
"""
import math
import argparse
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F
import csv


# Simple residual wrapper
class SkipConnection(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttentionLayer(nn.Module):
    """A standard Transformer-style attention + feed-forward block.

    Uses nn.MultiheadAttention internally for simplicity.
    Input shape: (batch, n_nodes, node_dim)
    """

    def __init__(self, embed_dim: int, n_heads: int, ff_hidden: int):
        super().__init__()
        self.project_in = nn.Linear(2, embed_dim)  # we'll project (x,y) -> embed
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.norm_attn = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_hidden)
        self.norm_ff = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        # x: (batch, n, 2) or (batch, n, embed_dim) if already projected
        if x.size(-1) == 2:
            h = self.project_in(x)
        else:
            h = x

        # Self-attention (batch_first=True: (batch, n, embed))
        attn_out, _ = self.attn(h, h, h, key_padding_mask=mask)
        h = self.norm_attn(h + attn_out)
        ff_out = self.ff(h)
        h = self.norm_ff(h + ff_out)
        return h


class GraphAttentionEncoder(nn.Module):
    """Graph attention encoder that returns node embeddings and graph embedding.

    Produces per-node embeddings; a small classification head maps each node to
    3 logits (assignment cost / routing cost / loss cost), then softmax to
    probabilities.
    """

    def __init__(
        self,
        n_heads: int = 4,
        embed_dim: int = 128,
        n_layers: int = 3,
        ff_hidden: int = 256,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [MultiHeadAttentionLayer(embed_dim, n_heads, ff_hidden) for _ in range(n_layers)]
        )
        # final projection to embedding dim (if first layer projects from XY)
        self.embed_dim = embed_dim
        self.node_proj = nn.Linear(embed_dim, embed_dim)
        # per-node head to produce 3 logits
        self.head = nn.Linear(embed_dim, 3)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: (batch, n_nodes, 2) coordinates (float)
        :return: (node_probs, node_embeddings)
          - node_probs: (batch, n_nodes, 3) softmax probabilities per node
          - node_embeddings: (batch, n_nodes, embed_dim)
        """
        h = x
        for layer in self.layers:
            h = layer(h, mask=mask)

        h = self.node_proj(h)
        logits = self.head(h)
        probs = F.softmax(logits, dim=-1)
        return probs, h


def read_coordinates(filename: str):
    """Read coordinates file with lines 'x,y' -> returns list of (x,y) floats"""
    coords = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0].strip())
                y = float(parts[1].strip())
            except ValueError:
                continue
            coords.append((x, y))
    return coords


def compute_and_save_probs(infile: str, outfile: str, device: str = 'cpu') -> str:
    """Reads coordinates, computes attention-based 3-state probabilities and saves CSV.

    Output CSV columns: x,y,p_assign,p_route,p_loss
    """
    coords = read_coordinates(infile)
    if not coords:
        raise RuntimeError(f'No coordinates read from {infile}')

    # Normalize coordinates to [0,1] using min/max for numerical stability
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    rng_x = maxx - minx if maxx > minx else 1.0
    rng_y = maxy - miny if maxy > miny else 1.0

    norm = [( (x - minx) / rng_x, (y - miny) / rng_y ) for x,y in coords]

    tensor = torch.tensor([norm], dtype=torch.float32, device=device)  # (1, N, 2)

    model = GraphAttentionEncoder(n_heads=4, embed_dim=128, n_layers=3).to(device)
    model.eval()
    with torch.no_grad():
        probs, embeddings = model(tensor)

    probs = probs.squeeze(0).cpu().numpy()  # (N,3)

    # Save CSV
    with open(outfile, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x','y','p_assign','p_route','p_loss'])
        for (x,y), p in zip(coords, probs):
            writer.writerow([x, y, float(p[0]), float(p[1]), float(p[2])])

    return outfile


def main():
    parser = argparse.ArgumentParser(description='Run attention encoder on coordinates and save 3-state probs')
    parser.add_argument('--input', '-i', default='tsp数据坐标集.txt', help='Input coordinates file (x,y per line)')
    parser.add_argument('--output', '-o', default='attention_probs.csv', help='Output CSV file')
    parser.add_argument('--device', '-d', default='cpu', help='Torch device')
    args = parser.parse_args()

    out = compute_and_save_probs(args.input, args.output, device=args.device)
    print(f'Saved attention probabilities to {out}')


if __name__ == '__main__':
    main()
            return input


# 实现完整的Transformer注意力 层！ 例如上方的组件进行注册
class MultiHeadAttentionLayer(nn.Sequential):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            normalization='batch',
    ):
        super(MultiHeadAttentionLayer, self).__init__(
            # 残差连接
            SkipConnection(
                MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
            ),
            # 归一化
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization)
        )


# 图注意力编码器 图的节点特征 编码为节点级、图级嵌入
class GraphAttentionEncoder(nn.Module):
    """
    Graph Attention Encoder that transforms input node features into embeddings
    using multi-head attention layers.
    
    Args:
        n_heads (int): Number of attention heads
        embed_dim (int): Embedding dimension
        n_layers (int): Number of attention layers
        node_dim (int, optional): Input node dimension
        normalization (str): Type of normalization ('batch' or 'instance')
        feed_forward_hidden (int): Hidden dimension in feed-forward layers
    """
    
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            node_dim=None,
            normalization='batch',
            feed_forward_hidden=512
    ):
        super(GraphAttentionEncoder, self).__init__()

        # To map input to embedding space
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None

        self.layers = nn.Sequential(*(
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization)
            for _ in range(n_layers)
        ))

    # 图级节点的编码和嵌入
    def forward(self, x, mask=None):
        """
        Forward pass through the encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, graph_size, node_dim)
            mask (torch.Tensor, optional): Attention mask
            
        Returns:
            tuple: 
                - h (torch.Tensor): Node embeddings (batch_size, graph_size, embed_dim)
                - graph_embedding (torch.Tensor): Graph-level embedding (batch_size, embed_dim)
        """
        assert mask is None, "TODO mask not yet supported!"

        # Batch multiply to get initial embeddings of nodes
        h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1) if self.init_embed is not None else x

        h = self.layers(h)


def export_torchscript(model: nn.Module, example_input: torch.Tensor, path: str):
        """
        Trace and save `model` as TorchScript for loading from C++ (libtorch).

        Example:
            model = SVRAPStateAttentionEncoder(...)
            example = torch.rand(1, N, node_dim)
            export_torchscript(model, example, 'svrap_state_encoder.pt')

        In C++ you can then load with `torch::jit::load("svrap_state_encoder.pt")`.
        """
        model.eval()
        with torch.no_grad():
                traced = torch.jit.trace(model, example_input)
                traced.save(path)
        return path


        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )
