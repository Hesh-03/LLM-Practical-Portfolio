"""
Chat window for the trained Tiny Transformer (Lab 03).

Loads the saved model (results/tinystories_gpt.pt) and starts a Gradio chat window.
No training is needed. Run from a terminal:

    conda activate llm-lab
    python chat_app.py

Then open http://127.0.0.1:7860 in your browser. Press Ctrl+C in the terminal to stop.
"""
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    # Masked multi-head self-attention
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)     # queries, keys, values in one matrix
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)         # output projection
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # reshape to (batch, heads, time, head_size)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))      # scaled dot-product
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))  # causal mask
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)             # concatenate heads
        return self.resid_drop(self.proj(y))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    # One Transformer block: LayerNorm -> attention -> residual, LayerNorm -> MLP -> residual
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.ffn = FeedForward(cfg)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)     # token embeddings
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)     # positional embeddings
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight                    # weight tying
        self.apply(self._init_weights)
        for name, p in self.named_parameters():                  # scaled init for residual projections
            if name.endswith("proj.weight") or name.endswith("net.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, "sequence longer than block_size"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, stop_id=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if stop_id is not None and next_id.item() == stop_id:
                break
        return idx


# ---------------------------------------------------------------- load the saved model
CHECKPOINT = Path(__file__).parent / "results" / "tinystories_gpt.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt = torch.load(CHECKPOINT, map_location=device)
chars = ckpt["chars"]
eos_id = ckpt["eos_id"]
EOS = chars[eos_id]
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
space_id = stoi[" "]
vocab_size = len(chars)

model = GPT(GPTConfig(**ckpt["config"])).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Loaded model from {CHECKPOINT.name}: {sum(p.numel() for p in model.parameters()):,} parameters, device = {device}")


def encode(text):
    return [stoi.get(c, space_id) for c in text]


def decode(ids):
    return "".join(itos[int(i)] for i in ids)


def generate_text(prompt, max_new_tokens=300, temperature=0.8, top_k=40):
    ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(ids, max_new_tokens, temperature=temperature, top_k=top_k, stop_id=eos_id)
    return decode(out[0].tolist()).replace(EOS, "\n[END OF STORY]")


# ---------------------------------------------------------------- Gradio chat window
def chat_fn(message, history, max_new_tokens, temperature, top_k):
    return generate_text(message, int(max_new_tokens), float(temperature), int(top_k))


if __name__ == "__main__":
    import gradio as gr

    demo = gr.ChatInterface(
        fn=chat_fn,
        title="Tiny Story Teller (character-level Transformer)",
        description="Type the start of a story, for example: Once upon a time",
        additional_inputs=[
            gr.Slider(50, 800, value=300, step=10, label="Max new characters"),
            gr.Slider(0.3, 1.5, value=0.8, step=0.05, label="Temperature"),
            gr.Slider(1, vocab_size, value=40, step=1, label="Top-k"),
        ],
    )
    demo.launch()
