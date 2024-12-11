from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
# train_data = np.load('./prompt/new_data/imgtxt_train_text_unprompt.npy')
# val_data = np.load('./prompt/new_data/imgtxt_val_text_unprompt.npy')
# dev_data = np.load('./prompt/new_data/imgtxt_test_text_unprompt.npy')

# train_data = np.load('./prompt/gpt/text_train_implied_gpt_new.npy')
# val_data = np.load('./prompt/gpt/text_val_implied_gpt_new.npy')
# dev_data = np.load('./prompt/gpt/text_test_implied_gpt_new.npy')

train_data = np.load('./prompt/new-llm/tweet_train_implied_new.npy')
val_data = np.load('./prompt/new-llm/tweet_val_implied_new.npy')
dev_data = np.load('./prompt/new-llm/tweet_test_implied_new.npy')

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased').to('cuda:0')
linear_layer = nn.Linear(768, 512).cuda()
encoded_val = []
index = 0
for text in tqdm(val_data, desc="Encoding texts", unit="text"):
    # 使用tokenizer对文本进行标记化
    tokens = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=30)
    tokens = tokens.to('cuda:0')
    # 使用模型进行前向传播
    with torch.no_grad():
        outputs = model(**tokens)
    # 获取输出中的 last_hidden_state（最后一层的隐藏状态）
    last_hidden_state = outputs.last_hidden_state
    out = last_hidden_state[:, 0, :]  # [batch, 768]
    out = linear_layer(out)
    out = out.cpu().detach().numpy()
    encoded_val.append(out)
# 将列表转换为NumPy数组
# torch.save(encoded_val, 'val_deberta_3d.pt')
encoded_val = np.array(encoded_val)
encoded_val = np.reshape(encoded_val, (5000,512))
np.save( './bert/flant5_val', encoded_val)
#
encoded_dev = []
index = 0
for text in tqdm(dev_data, desc="Encoding texts", unit="text"):
    # 使用tokenizer对文本进行标记化
    tokens = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=30)
    tokens = tokens.to('cuda:0')
    # 使用模型进行前向传播
    with torch.no_grad():
        outputs = model(**tokens)
    # 获取输出中的 last_hidden_state（最后一层的隐藏状态）
    last_hidden_state = outputs.last_hidden_state
    out = last_hidden_state[:, 0, :]  # [batch, 768]
    out = linear_layer(out)
    out = out.cpu().detach().numpy()
    encoded_dev.append(out)
# encoded_dev = [tensor.numpy() for tensor in encoded_dev]


# 将列表转换为NumPy数组
encoded_dev = np.array(encoded_dev)
encoded_dev = np.reshape(encoded_dev, (10000,512))
np.save( './bert/flant5_test.npy', encoded_dev)
# # 现在，encoded_texts 包含了每个文本对应的 DeBERTa 编码
# torch.save(encoded_dev, 'dev_deberta_3d.pt')

encoded_train = []
index = 0
for text in tqdm(train_data, desc="Encoding texts", unit="text"):
    # 使用tokenizer对文本进行标记化
    tokens = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=30)
    tokens = tokens.to('cuda:0')
    # 使用模型进行前向传播
    with torch.no_grad():
        outputs = model(**tokens)
    # 获取输出中的 last_hidden_state（最后一层的隐藏状态）
    last_hidden_state = outputs.last_hidden_state
    out = last_hidden_state[:, 0, :]  # [batch, 768]
    out = linear_layer(out)
    out = out.cpu().detach().numpy()
    encoded_train.append(out)

    # 将列表转换为NumPy数组
encoded_train = np.array(encoded_train)
encoded_train = np.reshape(encoded_train, (134823,512))
np.save('./bert/flant5_train.npy', encoded_train)
