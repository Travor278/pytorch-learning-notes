from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MiniMindConfig:
    vocab_size: int = 6400
    hidden_size: int = 768
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int | None = None
    intermediate_size: int | None = None
    max_position_embeddings: int = 32768
    rope_theta: float = 1e6
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0
    use_moe: bool = False
    num_experts: int = 4
    num_experts_per_tok: int = 1
    router_aux_loss_coef: float = 5e-4
    tie_word_embeddings: bool = True
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        if self.head_dim is None:
            if self.hidden_size % self.num_attention_heads != 0:
                raise ValueError("hidden_size must be divisible by num_attention_heads")
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

        if self.intermediate_size is None:
            self.intermediate_size = math.ceil(self.hidden_size * math.pi / 64) * 64

        if self.num_key_value_heads <= 0:
            raise ValueError("num_key_value_heads must be positive")


@dataclass
class MiniMindCausalLMOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor
    hidden_states: torch.Tensor
    aux_loss: torch.Tensor | None = None


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(dtype)


def precompute_rope_cache(
    head_dim: int,
    max_position_embeddings: int,
    rope_theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(max_position_embeddings, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    batch, num_kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


class MiniMindAttention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.dropout = config.dropout

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        attn_mask = self._build_attention_mask(
            attention_mask=attention_mask,
            batch_size=batch_size,
            seq_len=seq_len,
            dtype=q.dtype,
            device=q.device,
        )
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=attn_mask is None,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)
        return self.resid_dropout(attn_output)

    @staticmethod
    def _build_attention_mask(
        attention_mask: torch.Tensor | None,
        batch_size: int,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        if attention_mask is None:
            return None

        if attention_mask.dim() != 2:
            raise ValueError("attention_mask must have shape [batch, seq]")

        causal = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        )
        padding = attention_mask.to(device=device).eq(0).view(batch_size, 1, 1, seq_len)
        mask = causal.view(1, 1, seq_len, seq_len) | padding
        additive_mask = torch.zeros(batch_size, 1, seq_len, seq_len, dtype=dtype, device=device)
        return additive_mask.masked_fill(mask, torch.finfo(dtype).min)


class MiniMindMLP(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)
        return self.dropout(x)


class MiniMindMoE(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        if not 0 < self.num_experts_per_tok <= self.num_experts:
            raise ValueError("num_experts_per_tok must be in [1, num_experts]")
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([MiniMindMLP(config) for _ in range(config.num_experts)])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        original_shape = x.shape
        flat_x = x.reshape(-1, original_shape[-1])
        router_logits = self.gate(flat_x)
        router_probs = F.softmax(router_logits.float(), dim=-1)
        topk_weights, topk_indices = torch.topk(
            router_probs,
            k=self.num_experts_per_tok,
            dim=-1,
        )
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        topk_weights = topk_weights.to(dtype=flat_x.dtype)

        flat_output = torch.zeros_like(flat_x)
        for expert_idx, expert in enumerate(self.experts):
            selected = topk_indices.eq(expert_idx)
            if not bool(selected.any()):
                continue
            token_indices, slot_indices = selected.nonzero(as_tuple=True)
            expert_output = expert(flat_x[token_indices])
            expert_weight = topk_weights[token_indices, slot_indices].unsqueeze(-1)
            flat_output.index_add_(0, token_indices, expert_output * expert_weight)

        expert_load = F.one_hot(topk_indices, num_classes=self.num_experts).float().sum(dim=1)
        expert_load = expert_load.mean(dim=0) / self.num_experts_per_tok
        expert_importance = router_probs.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(expert_importance * expert_load)
        return flat_output.reshape(original_shape), aux_loss


class MiniMindDecoderLayer(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = MiniMindAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MiniMindMoE(config) if config.use_moe else MiniMindMLP(config)
        self.use_moe = config.use_moe

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        aux_loss = None
        if self.use_moe:
            hidden_states, aux_loss = self.mlp(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
        return residual + hidden_states, aux_loss


class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [MiniMindDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        cos, sin = precompute_rope_cache(
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
        )
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_position_embeddings:
            raise ValueError(
                f"seq_len={seq_len} exceeds max_position_embeddings="
                f"{self.config.max_position_embeddings}"
            )

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device)
        elif position_ids.dim() != 1:
            raise ValueError("position_ids must be a 1D tensor for the M1 Dense path")

        hidden_states = self.embed_tokens(input_ids)
        cos = self.cos_cached.index_select(0, position_ids).to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        sin = self.sin_cached.index_select(0, position_ids).to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        position_embeddings = (cos, sin)

        aux_losses: list[torch.Tensor] = []
        for layer in self.layers:
            hidden_states, aux_loss = layer(hidden_states, position_embeddings, attention_mask)
            if aux_loss is not None:
                aux_losses.append(aux_loss)

        hidden_states = self.norm(hidden_states)
        aux_loss = torch.stack(aux_losses).mean() if aux_losses else None
        return hidden_states, aux_loss


class MiniMindForCausalLM(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.model = MiniMindModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self._zero_pad_embedding()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def _zero_pad_embedding(self) -> None:
        if self.config.pad_token_id is None:
            return
        with torch.no_grad():
            self.model.embed_tokens.weight[self.config.pad_token_id].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> MiniMindCausalLMOutput:
        hidden_states, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        logits = self.lm_head(hidden_states)
        loss = None

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            if aux_loss is not None:
                loss = loss + self.config.router_aux_loss_coef * aux_loss

        return MiniMindCausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            aux_loss=aux_loss,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int | None = 50,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
        vocab_size_limit: int | None = None,
        suppress_token_ids: list[int] | tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [batch, seq]")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")

        eos_token_id = self.config.eos_token_id if eos_token_id is None else eos_token_id
        pad_token_id = self.config.pad_token_id if pad_token_id is None else pad_token_id
        was_training = self.training
        self.eval()

        generated = input_ids
        finished = torch.zeros(generated.size(0), dtype=torch.bool, device=generated.device)
        for _ in range(max_new_tokens):
            context = generated[:, -self.config.max_position_embeddings :]
            output = self(input_ids=context)
            next_token_logits = output.logits[:, -1, :].clone()

            if vocab_size_limit is not None:
                if not 0 < vocab_size_limit <= next_token_logits.size(-1):
                    raise ValueError("vocab_size_limit must be within the model vocabulary")
                next_token_logits[:, vocab_size_limit:] = -torch.inf

            if suppress_token_ids:
                valid_ids = [
                    token_id
                    for token_id in suppress_token_ids
                    if 0 <= token_id < next_token_logits.size(-1)
                ]
                if valid_ids:
                    next_token_logits[:, valid_ids] = -torch.inf

            if temperature == 0:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                next_token_logits = next_token_logits / temperature
                if top_k is not None and top_k > 0:
                    k = min(top_k, next_token_logits.size(-1))
                    threshold = torch.topk(next_token_logits, k, dim=-1).values[:, -1:]
                    next_token_logits = next_token_logits.masked_fill(
                        next_token_logits < threshold,
                        -torch.inf,
                    )
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            if eos_token_id is not None:
                replacement = torch.full_like(next_token, pad_token_id)
                next_token = torch.where(finished.unsqueeze(-1), replacement, next_token)
                finished = finished | next_token.squeeze(-1).eq(eos_token_id)

            generated = torch.cat([generated, next_token], dim=-1)
            if bool(finished.all()):
                break

        if was_training:
            self.train()
        return generated
