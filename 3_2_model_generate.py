import torch
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Tokenizer, GPT2LMHeadModel, GPT2Config

tokenizer = GPT2Tokenizer.from_pretrained("gpt2") # すでに学習済みのトークナイザを読み込む
# gpt2 = GPT2Model.from_pretrained("gpt2")          # すでに学習済みのモデルを読み込む
# gpt2.eval()
config = GPT2Config.from_pretrained("gpt2")       # モデルの設定を読み込む
gpt2_lm = GPT2LMHeadModel.from_pretrained("gpt2") # すでに学習済みのヘッドを読み込む
gpt2_lm.eval()


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

    for _ in range(max_new_tokens): # 最大 max_new_tokens 個のトークンを生成する
        with torch.no_grad():       # 勾配計算を無効化
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


inputs = tokenizer("The quick brown fox", return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

output_ids = generate(gpt2_lm,
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=20,
                temperature=1.0,
                do_sample=False,
                top_k=None,
                eos_token_id=tokenizer.eos_token_id)

output_ids = output_ids[0]
print(output_ids)
print(tokenizer.decode(output_ids, skip_special_tokens=True))

# tensor([[  464,  2068,  7586, 21831,   274,   389,   257,  1049,   835,   284,
#            651,   257,  1310,  1643,   286,   257,  4829,   503,   286,   534,
#           3290,    13,   198,   198]])
# The quick brown foxes are a great way to get a little bit of a kick out of your dog.