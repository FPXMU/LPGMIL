import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mamba.mamba_ssm import Mamba
def initialize_weights(module: nn.Module) -> None:
    """Initialize module weights using Xavier initialization for Linear layers."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class Classifier1FC(nn.Module):
    """Single-layer classifier with optional dropout."""
    def __init__(self, in_features: int, num_classes: int, dropout_rate: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        return self.fc(x)

class SingleLayerMLP(nn.Module):
    """Single-layer MLP projection module."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.1,
        num_masked_patches: int = 0,
        mask_drop_rate: float = 0.0
    ):
        super().__init__()
        self.num_masked_patches = num_masked_patches
        self.mask_drop_rate = mask_drop_rate
        self.inner_dim = embed_dim // downsample_rate
        self.num_heads = num_heads
        assert self.inner_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, self.inner_dim)
        self.k_proj = nn.Linear(embed_dim, self.inner_dim)
        self.v_proj = nn.Linear(embed_dim, self.inner_dim)
        self.out_proj = nn.Linear(self.inner_dim, embed_dim)

        self.layer_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def _reshape_heads(self, x: Tensor) -> Tensor:
        b, seq_len, _ = x.shape
        return x.view(b, seq_len, self.num_heads, -1).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        b, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(b, seq_len, -1)

    def _apply_masking(self, attn: Tensor) -> Tensor:
        if not self.training or self.num_masked_patches <= 0:
            return attn
        b, h, q, k = attn.shape
        num_mask = min(self.num_masked_patches, k)
        drop_n = int(num_mask * self.mask_drop_rate)
        if drop_n == 0:
            return attn
        _, topk_idx = attn.topk(num_mask, dim=-1)           
        flat_topk = topk_idx.view(-1, num_mask)             
        rand = torch.rand(flat_topk.shape, device=attn.device).argsort(dim=-1)
        drop_idx = flat_topk.gather(1, rand[:, :drop_n])    
        flat_mask = torch.ones((b*h*q, k), device=attn.device, dtype=torch.bool)
        rows = torch.arange(b*h*q, device=attn.device).unsqueeze(1).expand(-1, drop_n)
        flat_mask[rows, drop_idx] = False
        mask = flat_mask.view(b, h, q, k)
        return attn.masked_fill(~mask, -1e9)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.q_proj(q);  k = self.k_proj(k);  v = self.v_proj(v)
        q = self._reshape_heads(q)  
        k = self._reshape_heads(k)  
        v = self._reshape_heads(v) 
        attn_weights = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        attn_weights = self._apply_masking(attn_weights)
        raw_scores = attn_weights.clone()                                 
        attn_probs = torch.softmax(attn_weights, dim=-1)
        out = attn_probs @ v                   
        out = self._merge_heads(out)            
        out = self.out_proj(self.dropout(out))
        out = self.layer_norm(out)             
        feat = out.squeeze(1)                  
        attn  = raw_scores[0]                   
        return feat, attn
class Aggratt(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.inner_dim = embed_dim // downsample_rate
        self.num_heads = num_heads
        assert self.inner_dim % num_heads == 0, \
            f"Embedding dim ({self.inner_dim}) must be divisible by num_heads ({num_heads})"

        # projections
        self.v_proj = nn.Linear(embed_dim, self.inner_dim)
        self.out_proj = nn.Linear(self.inner_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, v: Tensor, attn_weights: Tensor) -> Tensor:

        if attn_weights.dim() == 3:
            attn_weights = attn_weights.unsqueeze(0) 
        
        B, heads, tgt_len, src_len = attn_weights.shape
        assert heads == self.num_heads, \
            f"Expected heads={self.num_heads}, got {heads}"
        if v.dim() == 2:
            v = v.unsqueeze(0)
        v_proj = self.v_proj(v) 
        head_dim = self.inner_dim // self.num_heads
        v_heads = (
            v_proj
            .view(B, src_len, self.num_heads, head_dim)  
            .transpose(1, 2)                             
        )
        weighted = torch.matmul(attn_weights, v_heads)
        combined = (
            weighted
            .transpose(1, 2)                            
            .contiguous()
            .view(B, tgt_len, self.inner_dim)            
        )
        out = self.out_proj(combined)  
        out = self.dropout(out)
        out = self.layer_norm(out)
        if tgt_len == 1:
            out = out.squeeze(1) 
        return out

class LPGMIL(nn.Module):
    def __init__(
        self,
        in_dim: int = 1024,
        num_classes: int = 2,
        dropout: float = 0.25,
        activation: str = 'relu',
        num_layers: int = 2,
        topk_rate: int = 10,
        num_masked_patches: int = 10,
        num_tokens: int = 6,
        mask_drop_rate: float = 0.6,
        model_type: str = "Mamba",
        use_topk: bool = True,
        prototypes: Tensor = None
    ):
        super().__init__()
        self.feature_extractor = self._build_feature_extractor(in_dim, dropout, activation)
        self.norm = nn.LayerNorm(512)
        self.mamba_layers = self._build_mamba_layers(num_layers) if model_type == "Mamba" else None
        self.projection = SingleLayerMLP(1024, 512)
        self.topk_selector = MultiHeadAttention(512, 8)
        self.token_attentions = nn.ModuleList([
            MultiHeadAttention(512, 8, num_masked_patches=num_masked_patches, mask_drop_rate=mask_drop_rate)
            for _ in range(num_tokens)
        ])
        self.bag_attention = Aggratt(512, 8)
        self.prototypes = self._init_prototypes(prototypes)
        self.prototype_selector = prototypes.unsqueeze(0) if use_topk else None
        self.token_classifiers = nn.ModuleList([
            Classifier1FC(512, num_classes) for _ in range(num_tokens)
        ])
        self.slide_classifier = Classifier1FC(512, num_classes)
        self.topk_rate = topk_rate
        self.num_tokens = num_tokens
        self.model_type = model_type
        
        initialize_weights(self)

    def _build_feature_extractor(self, in_dim: int, dropout: float, activation: str) -> nn.Sequential:
        layers = [nn.Linear(in_dim, 512)]
        if activation.lower() == 'relu':
            layers.append(nn.ReLU())
        elif activation.lower() == 'gelu':
            layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def _build_mamba_layers(self, num_layers: int) -> nn.ModuleList:
        layers = nn.ModuleList()
        for _ in range(num_layers):
            layers.append(nn.Sequential(
                nn.LayerNorm(512),
                Mamba(
                    d_model=512,
                    d_state=16,
                    d_conv=4,
                    expand=2,
                )
            ))
        return layers

    def _init_prototypes(self, prototypes: Tensor) -> nn.Parameter:
        if prototypes is not None:
            print("Initializing with TLS prototypes")
            return nn.Parameter(prototypes.unsqueeze(0))
        print("Initializing random prototypes")
        return nn.Parameter(torch.zeros(1, 6, 1024))

    def forward(self, x: Tensor) -> tuple:
        x = x.unsqueeze(0)
        batch_size, num_patches, _ = x.shape
        hidden = self.feature_extractor(x.float())
        # Process through Mamba layers
        if self.model_type == "Mamba":
            for layer in self.mamba_layers:
                residual = hidden
                hidden = layer[0](hidden)
                hidden = layer[1](hidden) + residual
        
        hidden = self.norm(hidden)
        # Prototype processing
        query = self.projection(self.prototype_selector)
        if num_patches > 1000:
            inst, attn = self.topk_selector(query, hidden, hidden)
            probs = torch.softmax(attn.squeeze(0), dim=-1)[0].mean(0)
            topk_probs, topk_indices = torch.topk(probs, 1000)
            topk_probs_selected = torch.index_select(hidden.squeeze(0), 0, topk_indices)
            newk = topk_probs_selected.unsqueeze(0)
            newv = topk_probs_selected.unsqueeze(0)
        token_outputs, attentions = [], []
        for idx in range(self.num_tokens):
            token_query = self.prototypes[:, idx:idx+1]
            if num_patches > 1000:
                token_feat, token_attn = self.token_attentions[idx](token_query, newk, newv)
            else:
                token_feat, token_attn = self.token_attentions[idx](token_query, hidden, hidden)
            
            token_outputs.append(self.token_classifiers[idx](token_feat))
            attentions.append(token_attn)
        
       
        combined_attn = torch.cat(attentions, dim=1)
        A_out = combined_attn
        context = newk if num_patches > 1000 else hidden
        bag_feature = self.bag_attention(context, combined_attn.softmax(dim=-1).mean(1, keepdim=True))
      
        logits = self.slide_classifier(bag_feature)
        Y_prob = F.softmax(logits, dim=-1)
        Y_hat = torch.argmax(logits, dim=-1)
        
        return logits, Y_prob, Y_hat, A_out, torch.stack(token_outputs), combined_attn

    def to_device(self, device: torch.device) -> None:
        """Move model to specified device."""
        self.to(device)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_input = torch.randn(2000, 1024).to(device)
    prototypes = torch.randn(6, 1024).to(device)
    model = LPGMIL(
        in_dim=1024,
        num_masked_patches=10,
        num_tokens=6,
        mask_drop_rate=0.6,
        prototypes=prototypes
    ).to(device)
    model(test_input)