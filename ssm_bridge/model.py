import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Union
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class VLMConfig:
    """Configuration for Vision-Language Model"""

    # Vision encoder
    img_size: int = 224
    patch_size: int = 16
    vision_embed_dim: int = 768
    vision_num_layers: int = 12
    vision_num_heads: int = 12

    # Language model
    vocab_size: int = 32000
    text_embed_dim: int = 768
    text_num_layers: int = 12
    text_num_heads: int = 12
    text_kv_heads: int = 1
    max_seq_length: int = 2048

    # SSM Bridge
    ssm_state_dim: int = 64
    ssm_hidden_dim: int = 256
    ssm_num_layers: int = 4
    ssm_dropout: float = 0.1

    # Cross-modal attention
    cross_attn_layers: int = 2
    cross_attn_heads: int = 8

    # Training
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5


# Enhanced Swish activation with better numerical stability
class Swish(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(torch.clamp(x, min=-10, max=10))


# Enhanced SwiGLU with residual connection and layer norm
class SwiGLU(nn.Module):
    def __init__(
        self, dim: int, hidden_dim: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim, eps=1e-5)
        self.swish = Swish()

        # Proper initialization
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.xavier_uniform_(self.w3.weight)
        nn.init.zeros_(self.w1.bias)
        nn.init.zeros_(self.w2.bias)
        nn.init.zeros_(self.w3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        gate = self.swish(self.w1(x))
        hidden = self.w2(x)
        output = self.w3(self.dropout(gate * hidden))
        return residual + output


# Enhanced Multi-Query Attention with better numerical stability
class MultiQueryAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        kv_heads: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert (
            num_heads % kv_heads == 0
        ), "num_heads must be divisible by kv_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.kv_heads = kv_heads
        self.head_dim = dim // num_heads
        self.q_heads_per_kv = num_heads // kv_heads

        self.q_proj = nn.Linear(dim, num_heads * self.head_dim)
        self.k_proj = nn.Linear(dim, kv_heads * self.head_dim)
        self.v_proj = nn.Linear(dim, kv_heads * self.head_dim)
        self.o_proj = nn.Linear(num_heads * self.head_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Proper initialization
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.zeros_(self.o_proj.bias)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, C = x.shape
        # Queries: [B, T, num_heads * head_dim]
        q = (
            self.q_proj(x)
            .view(B, T, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        # Keys and Values: [B, T, kv_heads * head_dim]
        k = (
            self.k_proj(x)
            .view(B, T, self.kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(B, T, self.kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Repeat K and V for each query head in the group
        k = k.repeat_interleave(self.q_heads_per_kv, dim=1)
        v = v.repeat_interleave(self.q_heads_per_kv, dim=1)

        # Attention scores with numerical stability
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = torch.clamp(scores, min=-100, max=100)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Output: [B, num_heads, T, head_dim] -> [B, T, num_heads * head_dim]
        out = (
            torch.matmul(attn, v)
            .transpose(1, 2)
            .contiguous()
            .view(B, T, -1)
        )
        return self.o_proj(out)


# Cross-modal attention for vision-language fusion
class CrossModalAttention(nn.Module):
    def __init__(
        self, dim: int, num_heads: int = 8, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, num_heads * self.head_dim)
        self.k_proj = nn.Linear(dim, num_heads * self.head_dim)
        self.v_proj = nn.Linear(dim, num_heads * self.head_dim)
        self.o_proj = nn.Linear(num_heads * self.head_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Proper initialization
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.zeros_(self.o_proj.bias)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T_q, C = query.shape
        _, T_k, _ = key.shape

        # Project to Q, K, V
        q = (
            self.q_proj(query)
            .view(B, T_q, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(key)
            .view(B, T_k, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(value)
            .view(B, T_k, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = torch.clamp(scores, min=-100, max=100)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Output
        out = (
            torch.matmul(attn, v)
            .transpose(1, 2)
            .contiguous()
            .view(B, T_q, -1)
        )
        return self.o_proj(out)


# Enhanced Transformer Block with residual connections and layer norm
class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        kv_heads: int = 1,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn = MultiQueryAttention(
            dim, num_heads, kv_heads, dropout
        )
        self.ffn = SwiGLU(dim, ffn_dim, dropout)
        self.ln1 = nn.LayerNorm(dim, eps=1e-5)
        self.ln2 = nn.LayerNorm(dim, eps=1e-5)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with residual
        residual = x
        x = self.ln1(x)
        x = self.attn(x, mask)
        x = self.dropout(x)
        x = residual + x

        # FFN with residual
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        return x


# Enhanced SSM Bridge with multiple layers and better state management
class SSMLayer(nn.Module):
    """Single SSM layer as a proper nn.Module"""

    def __init__(self, hidden_dim: int, state_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim

        # Input projection to state dimension
        self.input_proj = nn.Linear(hidden_dim, state_dim)

        # State transition parameters
        self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.1)
        self.B = nn.Parameter(torch.randn(state_dim) * 0.1)
        self.C = nn.Parameter(
            torch.randn(hidden_dim, state_dim) * 0.1
        )
        self.bias_A = nn.Parameter(torch.zeros(state_dim))
        self.bias_C = nn.Parameter(torch.zeros(hidden_dim))

        # Gating and projection layers
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

        # Initialize all layers
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.gate.bias)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, hidden_dim]
        B, T, _ = x.shape

        # Project input to state dimension
        x_state = self.input_proj(x)  # [B, T, state_dim]

        # Initialize state
        h = torch.zeros(B, self.state_dim, 1, device=x.device)
        outputs = []

        # Process sequence through SSM
        for t in range(T):
            x_t = x_state[:, t, :].unsqueeze(-1)  # [B, state_dim, 1]

            # State update with gating
            gate = torch.sigmoid(
                self.gate(x[:, t, :]).unsqueeze(-1)
            )  # Use original hidden_dim for gate
            h_new = (
                torch.matmul(self.A, h)
                + self.B.unsqueeze(-1) * x_t
                + self.bias_A.unsqueeze(-1)
            )
            h_new = torch.clamp(h_new, min=-10, max=10)
            h = h_new

            # Output projection with gating
            y_t = (
                torch.matmul(self.C, h).squeeze(-1) + self.bias_C
            )  # [B, hidden_dim]
            y_t = self.proj(y_t)
            y_t = (
                gate.squeeze(-1) * y_t
                + (1 - gate.squeeze(-1)) * x[:, t, :]
            )
            outputs.append(y_t)

        # Stack outputs
        return torch.stack(outputs, dim=1)


class EnhancedSSMBridge(nn.Module):
    def __init__(
        self,
        input_dim: int,
        state_dim: int = 64,
        hidden_dim: int = 256,
        output_dim: int = 768,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Input projection
        self.in_proj = nn.Linear(input_dim, hidden_dim)

        # Multiple SSM layers
        self.ssm_layers = nn.ModuleList(
            [
                SSMLayer(hidden_dim, state_dim)
                for _ in range(num_layers)
            ]
        )

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, output_dim)

        # Layer normalization and dropout
        self.layer_norms = nn.ModuleList(
            [
                nn.LayerNorm(hidden_dim, eps=1e-5)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

        # Proper initialization
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, input_dim]
        x = self.in_proj(x)  # [B, T, hidden_dim]

        # Process through SSM layers
        for i, (ssm_layer, ln) in enumerate(
            zip(self.ssm_layers, self.layer_norms)
        ):
            x = ln(x)
            x_new = ssm_layer(x)
            x = x + self.dropout(x_new)

        # Final output projection
        return self.out_proj(x)


# Enhanced Vision Transformer
class EnhancedVisionTransformer(nn.Module):
    def __init__(self, config: VLMConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.embed_dim = config.vision_embed_dim
        num_patches = (config.img_size // config.patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            3,
            config.vision_embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches + 1, config.vision_embed_dim)
            * 0.02
        )
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, config.vision_embed_dim) * 0.02
        )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.vision_embed_dim,
                    config.vision_num_heads,
                    ffn_dim=config.vision_embed_dim * 4,
                    dropout=config.dropout,
                )
                for _ in range(config.vision_num_layers)
            ]
        )

        self.ln = nn.LayerNorm(
            config.vision_embed_dim, eps=config.layer_norm_eps
        )
        self.dropout = nn.Dropout(config.dropout)

        # Proper initialization
        nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.zeros_(self.patch_embed.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(
            x
        )  # [B, embed_dim, H/patch_size, W/patch_size]
        x = x.flatten(2).transpose(
            1, 2
        )  # [B, num_patches, embed_dim]

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.dropout(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        return self.ln(x)


# Enhanced Vision-Language Model
class EnhancedVLM(nn.Module):
    def __init__(self, config: VLMConfig) -> None:
        super().__init__()
        self.config = config

        # Vision encoder
        self.vit = EnhancedVisionTransformer(config)

        # Enhanced SSM bridge
        self.ssm_bridge = EnhancedSSMBridge(
            input_dim=config.vision_embed_dim,
            state_dim=config.ssm_state_dim,
            hidden_dim=config.ssm_hidden_dim,
            output_dim=config.text_embed_dim,
            num_layers=config.ssm_num_layers,
            dropout=config.ssm_dropout,
        )

        # Language embedding
        self.token_embed = nn.Embedding(
            config.vocab_size, config.text_embed_dim
        )
        self.pos_embed = nn.Parameter(
            torch.randn(
                1, config.max_seq_length, config.text_embed_dim
            )
            * 0.02
        )

        # Cross-modal attention layers
        self.cross_attn_layers = nn.ModuleList(
            [
                CrossModalAttention(
                    config.text_embed_dim,
                    config.cross_attn_heads,
                    config.dropout,
                )
                for _ in range(config.cross_attn_layers)
            ]
        )

        # Language transformer
        self.text_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.text_embed_dim,
                    config.text_num_heads,
                    config.text_kv_heads,
                    ffn_dim=config.text_embed_dim * 4,
                    dropout=config.dropout,
                )
                for _ in range(config.text_num_layers)
            ]
        )

        self.ln = nn.LayerNorm(
            config.text_embed_dim, eps=config.layer_norm_eps
        )
        self.head = nn.Linear(
            config.text_embed_dim, config.vocab_size
        )
        self.dropout = nn.Dropout(config.dropout)

        # Proper initialization
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        images: torch.Tensor,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T = tokens.shape

        # Vision embeddings
        vision_embeds = self.vit(
            images
        )  # [B, num_patches+1, vision_embed_dim]

        # SSM bridge
        vision_embeds = self.ssm_bridge(
            vision_embeds
        )  # [B, num_patches+1, text_embed_dim]

        # Language embeddings
        token_embeds = self.token_embed(
            tokens
        )  # [B, T, text_embed_dim]
        token_embeds = token_embeds + self.pos_embed[:, :T, :]
        token_embeds = self.dropout(token_embeds)

        # Cross-modal attention
        for cross_attn in self.cross_attn_layers:
            token_embeds = cross_attn(
                token_embeds, vision_embeds, vision_embeds
            )

        # Concatenate vision and language embeddings
        x = torch.cat([vision_embeds, token_embeds], dim=1)

        # Causal mask for language tokens
        total_len = x.shape[1]
        mask = torch.tril(
            torch.ones(total_len, total_len, device=x.device)
        ).unsqueeze(0)

        # Apply language transformer blocks
        for block in self.text_blocks:
            x = block(x, mask)

        x = self.ln(x)
        logits = self.head(x[:, vision_embeds.shape[1] :, :])

        if targets is None:
            return logits

        # Compute loss with numerical stability
        logits = torch.clamp(logits, min=-100, max=100)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1)
        )
        return logits, loss

    def generate(
        self,
        images: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """Enhanced text generation with better sampling"""
        self.eval()
        B = images.shape[0]
        device = images.device

        # Start with BOS token
        tokens = torch.zeros(B, 1, dtype=torch.long, device=device)

        with torch.no_grad():
            for _ in range(max_length - 1):
                logits = self(images, tokens)[:, -1, :]

                # Apply temperature
                logits = logits / max(temperature, 1e-8)

                # Apply top-k filtering
                if top_k is not None:
                    top_k_logits, top_k_indices = torch.topk(
                        logits, min(top_k, logits.size(-1))
                    )
                    logits = torch.full_like(logits, float("-inf"))
                    logits.scatter_(-1, top_k_indices, top_k_logits)

                # Apply top-p (nucleus) sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        logits, descending=True
                    )
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = (
                        cumulative_probs > top_p
                    )
                    sorted_indices_to_remove[..., 1:] = (
                        sorted_indices_to_remove[..., :-1].clone()
                    )
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = (
                        sorted_indices_to_remove.scatter(
                            1,
                            sorted_indices,
                            sorted_indices_to_remove,
                        )
                    )
                    logits[indices_to_remove] = float("-inf")

                # Ensure numerical stability
                logits = torch.clamp(logits, min=-100, max=100)

                # Sample next token
                probs = F.softmax(logits, dim=-1)

                # Handle NaN or inf values
                if (
                    torch.isnan(probs).any()
                    or torch.isinf(probs).any()
                ):
                    probs = torch.ones_like(probs) / probs.size(-1)

                next_token = torch.multinomial(probs, num_samples=1)
                tokens = torch.cat([tokens, next_token], dim=1)

                # Stop if all sequences have EOS token
                if (next_token == 1).all():
                    break

        return tokens


# Example usage with enhanced configuration
if __name__ == "__main__":
    # Create configuration
    config = VLMConfig(
        img_size=224,
        patch_size=16,
        vision_embed_dim=768,
        vision_num_layers=12,
        vision_num_heads=12,
        vocab_size=32000,
        text_embed_dim=768,
        text_num_layers=12,
        text_num_heads=12,
        text_kv_heads=1,
        max_seq_length=2048,
        ssm_state_dim=64,
        ssm_hidden_dim=256,
        ssm_num_layers=4,
        ssm_dropout=0.1,
        cross_attn_layers=2,
        cross_attn_heads=8,
        dropout=0.1,
        layer_norm_eps=1e-5,
    )

    # Set device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Create enhanced model
    model = EnhancedVLM(config).to(device)

    # Create sample data
    images = torch.randn(2, 3, 224, 224).to(device)
    tokens = torch.randint(0, 32000, (2, 10)).to(device)
    targets = torch.randint(0, 32000, (2, 10)).to(device)

    # Test forward pass
    logits, loss = model(images, tokens, targets)
    print(
        f"Enhanced VLM - Logits shape: {logits.shape}, Loss: {loss.item():.4f}"
    )

    # # Print model summary
    # total_params = sum(p.numel() for p in model.parameters())
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Total parameters: {total_params:,}")
    # print(f"Trainable parameters: {trainable_params:,}")

    # # Test generation
    # try:
    #     generated_tokens = model.generate(images, max_length=20, temperature=0.8, top_k=50)
    #     print(f"Generated tokens shape: {generated_tokens.shape}")
    # except Exception as e:
    #     print(f"Generation failed: {e}")
