import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutput
from transformers import GPT2Model, GPT2Tokenizer, GPT2LMHeadModel, GPT2Config

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2 = GPT2Model.from_pretrained("gpt2")
gpt2.eval()
config = GPT2Config.from_pretrained("gpt2")
gpt2_lm = GPT2LMHeadModel.from_pretrained("gpt2") # すでに学習済みのヘッドを読み込む
gpt2_lm.eval()


class GPT2LMHeadLikeModel(nn.Module):
    """
    GPT2LMHeadのようなモデルを定義するクラス
    Headとは、モデルの出力層のこと。Baseモデルの出力（文章を意味が凝縮された行列）を受け取って、最終的な出力を計算する層。
    具体的には、GPT2LMHeadは、GPT2Modelの出力を受け取って、次のトークンの確率分布を計算する。
    """
    def __init__(self, base_model, vocab_size):
        super().__init__()
        self.gpt2 = base_model
        self.lm_head = nn.Linear(base_model.config.n_embd, vocab_size, bias=False) # ロジットを計算するための線形層

    def forward(self, input_ids, attention_mask=None, **kwargs):
        """
        Args:
            input_ids: (batch_size, sequence_length)
            attention_mask: (batch_size, sequence_length)
            kwargs: その他の引数
        Returns:
            outputs: モデルの出力
        """
        outputs = self.gpt2(input_ids, attention_mask=attention_mask, **kwargs) # モデルの出力
        hidden_states = outputs.last_hidden_state # 隠れ状態を取得
        logits = self.lm_head(hidden_states) # ロジットを計算

        return CausalLMOutput(logits=logits)


def generate(model, input_ids, attention_mask=None, max_new_tokens=30, temperature=1.0,
             do_sample=False, top_k=None, eos_token_id=None):
    """
    Args:
        model: ベースとするモデル
        input_ids: (batch_size, sequence_length)
        attention_mask: (batch_size, sequence_length)
        max_new_tokens: 生成するトークンの最大数
        temperature: 温度パラメータ
        do_sample: サンプリングをするか
        top_k: top-kサンプリングのk
        eos_token_id: 終了トークンのID
    Returns:
        generated: 生成されたトークン
    """
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        with torch.no_grad():   # 勾配計算を無効化
            outputs = model(input_ids=generated, attention_mask=attention_mask) # モデルの出力
            next_token_logits = outputs.logits[:, -1, :] / temperature
            if do_sample: # サンプリングをするか
                if top_k is not None:
                    # top-kサンプリング
                    # 上位kのトークンからサンプリング
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    # 上位kのトークン以外の確率を-Infにする
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
                # 確率分布を計算
                probs = F.softmax(next_token_logits, dim=-1)
                # 確率分布からトークンをサンプリング
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                # 貪欲生成 常に確率が最大のトークンを選択
                idx_next = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # 末尾に生成されたトークンを追加
        generated = torch.cat([generated, idx_next], dim=-1)

        # 終了トークンが出たら打ち切る
        if eos_token_id is not None:
            if torch.all(idx_next == eos_token_id):
                break

    return generated


my_lm_model = GPT2LMHeadLikeModel(gpt2, config.vocab_size)
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

# tensor([[  464,  2068,  7586, 21831,   274,   389,   257,  1049,   835,   284,
#            651,   257,  1310,  1643,   286,   257,  4829,   503,   286,   534,
#           3290,    13,   198,   198]])
# The quick brown foxes are a great way to get a little bit of a kick out of your dog.