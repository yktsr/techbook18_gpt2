import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, PreTrainedModel, GPT2Model, GPT2Tokenizer, GPT2LMHeadModel
from transformers.pytorch_utils import Conv1D
from transformers.modeling_outputs import CausalLMOutput
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions


class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.n_embd  # 埋め込み次元
        self.num_heads = config.n_head  # ヘッド数
        self.head_dim = self.embed_dim // self.num_heads  # 各ヘッドの次元

        self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim) # クエリ、キー、バリュー用の一括線形変換
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim) # 出力の線形変換
        self.split_size = self.embed_dim

    def forward(self, x, attention_mask=None):
        B, T, C = x.size()  # (バッチサイズ, シーケンス長, 埋め込み次元)
        query_states, key_states, value_states = self.c_attn(x).split(self.split_size, dim=2)

        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        shape_kv = (*key_states.shape[:-1], -1, self.head_dim)

        q = query_states.view(shape_q).transpose(1, 2)   # (B, heads, T, head_dim)
        k = key_states.view(shape_kv).transpose(1, 2)    # (B, heads, T, head_dim)
        v = value_states.view(shape_kv).transpose(1, 2)  # (B, heads, T, head_dim)

        # Scaled Dot-Product Attentionを計算
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)

        mask_value = torch.finfo(attn_weights.dtype).min
        attn_weights = attn_weights.masked_fill(causal_mask == 0, mask_value)

        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float("-inf"))

        attn_probs = F.softmax(attn_weights, dim=-1)  # (B, heads, T, T)
        attn_output = torch.matmul(attn_probs, v)  # (B, heads, T, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)

        output = self.c_proj(attn_output)  # (B, T, C)

        return output


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = Conv1D(4 * config.n_embd, config.n_embd)
        self.gelu    = nn.GELU()
        self.c_proj  = Conv1D(config.n_embd, 4 * config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class GPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = MaskedMultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln_1(x), attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x, None


class GPT2LikeModel(PreTrainedModel):
    def __init__(self, config: GPT2Config):
        super().__init__(config)

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(self, input_ids, position_ids=None, attention_mask=None, **kwargs):
        if position_ids is None:
            position_ids = torch.arange(0,
                                        input_ids.size(1),
                                        dtype=torch.long,
                                        device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        input_embeds = self.wte(input_ids)
        pos_embeds = self.wpe(position_ids)
        hidden_states = self.drop(input_embeds + pos_embeds)

        for block in self.h:
            hidden_states = block(hidden_states)[0]

        hidden_states = self.ln_f(hidden_states)

        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states)


class GPT2LMHeadLikeModel(nn.Module):
    def __init__(self, base_model, vocab_size):
        super().__init__()
        self.gpt2 = base_model
        self.lm_head = nn.Linear(config.n_embd, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.gpt2(input_ids, attention_mask=attention_mask, **kwargs) # モデルの出力
        hidden_states = outputs.last_hidden_state # 隠れ状態を取得
        logits = self.lm_head(hidden_states) # ロジットを計算

        return CausalLMOutput(logits=logits)


def generate(model, input_ids, attention_mask=None, max_new_tokens=30, temperature=1.0,
             do_sample=False, top_k=None, eos_token_id=None):
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        with torch.no_grad():   # 勾配計算を無効化
            outputs = model(input_ids=generated, attention_mask=attention_mask) # モデルの出力
            next_token_logits = outputs.logits[:, -1, :] / temperature
            if do_sample: # サンプリングをするか
                if top_k is not None:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(next_token_logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        generated = torch.cat([generated, idx_next], dim=-1)

        if eos_token_id is not None:
            if torch.all(idx_next == eos_token_id):
                break

    return generated


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2 = GPT2Model.from_pretrained("gpt2")
gpt2.eval()
config = GPT2Config.from_pretrained("gpt2")
gpt2_lm = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_lm.eval()

config.bias = False
config.dropout = 0.0

my_gpt2 = GPT2LikeModel(config)
my_gpt2.eval()
my_gpt2.load_state_dict(gpt2.state_dict()) # 各層の重みをまとめてすべてコピー
my_lm_model = GPT2LMHeadLikeModel(my_gpt2, config.vocab_size)
my_lm_model.eval()
my_lm_model.lm_head.weight.data = gpt2_lm.lm_head.weight.data.clone()

inputs = tokenizer("The quick brown fox", return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

text = generate(my_lm_model,
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=20,
                temperature=1.0,
                do_sample=False,
                top_k=None,
                eos_token_id=tokenizer.eos_token_id)

print(text)
print(tokenizer.decode(text[0], skip_special_tokens=True))