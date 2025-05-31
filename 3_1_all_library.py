from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# モデルとトークナイザの読み込み
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# 学習済みの重みをHugging Faceから自動で取得して読み込む
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# プロンプト
prompt = "The quick brown fox"
max_new_tokens = 20
temperature = 1.0

inputs = tokenizer(prompt, return_tensors="pt")
                                   # [  The, quick, brown,   fox]
input_ids = inputs["input_ids"]    # [  464,  2068,  7586, 21831]
attention_mask = inputs["attention_mask"] # [1, 1, 1, 1, 1, 1, 1, 1, 1]

with torch.no_grad():
    output_ids = model.generate(
        input_ids,                       # 入力ID
        attention_mask=attention_mask,   # 注意マスク
        max_new_tokens=max_new_tokens,   # 生成するトークン数
        temperature=temperature,         # 温度パラメータ（1.0はデフォルト）
        do_sample=False,                 # 確率的サンプリング　Falseの場合は貪欲生成
        top_k=50,                        # トップK制限（任意）
        pad_token_id=tokenizer.eos_token_id
    )

output_ids = output_ids[0]
# 出力ID
# [  The, quick, brown,   fox,
# [  464,  2068,  7586, 21831,

#     es,   are,   a,   great,   way,    to,
#    274,   389,   257,  1049,   835,   284,

#    get,     a,little,   bit,    of,     a,
#    651,   257,  1310,  1643,   286,   257,

#    kick,  out,    of,  your,   dog,      .             ]
#   4829,   503,   286,   534,  3290,    13,   198,   198]

# 生成されたトークン列（ID）をデコードしてテキストに変換
text = tokenizer.decode(output_ids, skip_special_tokens=True)

print(text)
# 出力: The quick brown foxes are a great way to get a little bit of a kick out of your dog.