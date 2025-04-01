import logging
import torch
import torch.nn as nn
from typing import Optional
from transformers.models.opt.modeling_opt import (
    OPTAttention,
    OPTConfig,
    OPTDecoderLayer,
    OPTDecoder,
    OPTModel,
    OPTForCausalLM,
)

logger = logging.getLogger(__name__)


class CustomLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_clusters: int, cluster_dim: int, bias: bool = True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.deficiency = out_features % cluster_dim
        if self.deficiency > 0:
            self.deficiency = cluster_dim - self.deficiency

        index_length = in_features * (out_features + self.deficiency) // cluster_dim
        self.cluster = nn.Parameter(torch.empty((num_clusters, cluster_dim), **factory_kwargs))
        index = torch.empty((index_length,), dtype=torch.int32, device=device) # save_pretraied doesn't support uint16
        self.register_buffer('index', index)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        vectors = self.cluster[self.index]
        if self.deficiency > 0:
            weight = vectors.view(self.in_features, -1)[:, :-self.deficiency]
        else:
            weight = vectors.view(self.in_features, -1)
            
        if self.bias is not None:
            out = torch.matmul(x, weight) + self.bias
        else:
            out = torch.matmul(x, weight)
        return out

        
class CustomOPTAttention(OPTAttention):
    def __init__(self, config: OPTConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.enable_bias = config.enable_bias
        self.num_clusters = config.num_clusters
        self.cluster_dim = config.cluster_dim
        self.k_proj = CustomLinear(self.embed_dim, self.embed_dim, self.num_clusters, self.cluster_dim, bias=self.enable_bias)
        self.v_proj = CustomLinear(self.embed_dim, self.embed_dim, self.num_clusters, self.cluster_dim, bias=self.enable_bias)
        self.q_proj = CustomLinear(self.embed_dim, self.embed_dim, self.num_clusters, self.cluster_dim, bias=self.enable_bias)
        self.out_proj = CustomLinear(self.embed_dim, self.embed_dim, self.num_clusters, self.cluster_dim, bias=self.enable_bias)

class CustomOPTDecoderLayer(OPTDecoderLayer):
    def __init__(self, config: OPTConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.embed_dim = config.hidden_size
        self.num_clusters = config.num_clusters
        self.cluster_dim = config.cluster_dim
        self.self_attn = CustomOPTAttention(config=config, layer_idx=layer_idx)
        self.fc1 = CustomLinear(self.embed_dim, config.ffn_dim, self.num_clusters, self.cluster_dim, bias=config.enable_bias)

class CustomOPTDecoder(OPTDecoder):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([CustomOPTDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)])

class CustomOPTModel(OPTModel):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.decoder = CustomOPTDecoder(config)

class CustomOPTForCausalLM(OPTForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = CustomOPTModel(config)