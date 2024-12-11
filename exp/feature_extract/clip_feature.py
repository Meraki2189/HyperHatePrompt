import os
import clip
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
torch.cuda.empty_cache()
# Load the model

def get_image_features(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).to('cuda:0')
    with torch.no_grad():
        encoding = model.encode_image(input_tensor.unsqueeze(0))
    return encoding

_tokenizer = _Tokenizer()
def tokenize(text, context_length: int = 77):

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    # all_tokens = _tokenizer.encode(text)
    tokens = [sot_token] + _tokenizer.encode(text)[:context_length-2] + [eot_token]
    result = torch.zeros(context_length, dtype=torch.long)
    mask = torch.zeros(context_length, dtype=torch.long)
    result[:len(tokens)] = torch.tensor(tokens)
    mask[:len(tokens)] = 1

    return result, mask
def process_and_save(data, save_path, max_token_length=77):
    batch_size = 16  # 根据你的GPU内存容量调整这个值
    with torch.no_grad():
        features = []
        total_batches = (len(data) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(data), batch_size), total=total_batches, desc="Processing batches"):
            batch = data[i:i + batch_size]
            batch = [text[:max_token_length] for text in batch]
            # batch = clip.tokenize(batch).to(device)
            result = [tokenize(text) for text in batch]
            # batch = [item[0] for item in result]
            batch = torch.stack([item[0] for item in result]).to(device)
            # print(batch.size())
            encoded_batch = model.encode_text(batch)
            # print(encoded_batch.size())
            features.append(encoded_batch.cpu().numpy())
            # 释放GPU内存
            torch.cuda.empty_cache()
    features = np.concatenate(features, axis=0)
    np.save(save_path, features)

#tweet
# train_tweet = np.load('prompt/new_data/tweet_train_text_unprompt.npy').tolist()
# val_tweet = np.load('prompt/new_data/tweet_val_text_unprompt.npy').tolist()
# test_tweet = np.load('prompt/new_data/tweet_test_text_unprompt.npy').tolist()
#tweet+prompt gpt
# train_tweet = np.load('prompt/gpt/tweet_train_text_prompt.npy').tolist()
# val_tweet = np.load('prompt/gpt/tweet_val_text_prompt.npy').tolist()
# test_tweet = np.load('prompt/gpt/tweet_test_text_prompt.npy').tolist()
#prompt gpt
# train_tweet = np.load('prompt/gpt/text_train_implied_gpt_new.npy').tolist()
# val_tweet = np.load('prompt/gpt/text_val_implied_gpt_new.npy').tolist()
# test_tweet = np.load('prompt/gpt/text_test_implied_gpt_new.npy').tolist()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device=device)

train_tweet = np.load('prompt/gpt/text_train_implied_gpt_new.npy').tolist()
val_tweet = np.load('prompt/gpt/text_val_implied_gpt_new.npy').tolist()
test_tweet = np.load('prompt/gpt/text_test_implied_gpt_new.npy').tolist()
process_and_save(val_tweet, 'clip/gpt_val.npy')
process_and_save(test_tweet, 'clip/gpt_test.npy')
process_and_save(train_tweet, 'clip/gpt_train.npy')

val_tweet = np.load('prompt/new_data/tweet_val_text_unprompt.npy').tolist()
test_tweet = np.load('prompt/new_data/tweet_test_text_unprompt.npy').tolist()
train_tweet = np.load('prompt/new_data/tweet_train_text_unprompt.npy').tolist()

process_and_save(val_tweet, 'clip/tweet_val.npy')
process_and_save(test_tweet, 'clip/tweet_test.npy')
process_and_save(train_tweet, 'clip/tweet_train.npy')

train_imgtxt = np.load('prompt/new_data/imgtxt_train_text_unprompt.npy').tolist()
val_imgtxt = np.load('prompt/new_data/imgtxt_val_text_unprompt.npy').tolist()
test_imgtxt = np.load('prompt/new_data/imgtxt_test_text_unprompt.npy').tolist()

process_and_save(train_imgtxt, 'clip/imgtxt_train.npy')
process_and_save(val_imgtxt, 'clip/imgtxt_val.npy')
process_and_save(test_imgtxt, 'clip/imgtxt_test.npy')

#flant5
train_flant5 = np.load('prompt/new-llm/tweet_train_implied_new.npy').tolist()
val_flant5 = np.load('prompt/new-llm/tweet_val_implied_new.npy').tolist()
test_flant5 = np.load('prompt/new-llm/tweet_test_implied_new.npy').tolist()

process_and_save(train_flant5, 'clip/flant5_train.npy')
process_and_save(val_flant5, 'clip/flant5_val.npy')
process_and_save(test_flant5, 'clip/flant5_test.npy')

feature_file2 = 'clip/image_val.npy'
val_keys = open('./dataset1/splits/val_ids.txt')
val_keys = val_keys.read()
val_keys = val_keys.splitlines()
i = 0
path1 = Path('./dataset1/img_resized')  # 图片
# 遍历文件夹中的所有图像文件，并提取特征向量
features2 = []
for keys in tqdm(val_keys):
    path_im = ""
    a = ""
    a = a.join([keys, ".jpg"])
    path_im = os.path.join(path1, a)
    img_t = preprocess(Image.open(path_im).convert('RGB')).unsqueeze(0).to(device)
    feature = model.encode_image(img_t).cpu().detach().numpy().tolist()
    features2.append(feature)
np.save(feature_file2, features2)
with open(feature_file2, 'rb') as file:
    loaded_data = np.load(file)
print(loaded_data.shape)
print('pass2')

feature_file3 = 'clip/image_test.npy'
test_keys = open('./dataset1/splits/test_ids.txt')
test_keys = test_keys.read()
test_keys = test_keys.splitlines()
i = 0
path1 = Path('./dataset1/img_resized')  # 图片
# 遍历文件夹中的所有图像文件，并提取特征向量
features3 = []
for keys in tqdm(test_keys):
    path_im = ""
    a = ""
    a = a.join([keys, ".jpg"])
    path_im = os.path.join(path1, a)
    img_t = preprocess(Image.open(path_im).convert('RGB')).unsqueeze(0).to(device)
    model = model.to('cuda')
    feature = model.encode_image(img_t.to('cuda')).cpu().detach().numpy().tolist()
    features3.append(feature)
np.save(feature_file3, features3)
with open(feature_file3, 'rb') as file:
    loaded_data = np.load(file)
print(loaded_data.shape)
print('pass3')

feature_file1 = 'clip/image_train.npy'
train_keys = open('./dataset1/splits/train_ids.txt')
train_keys = train_keys.read()
train_keys = train_keys.splitlines()
i = 0
path1 = Path('./dataset1/img_resized')  # 图片
# 遍历文件夹中的所有图像文件，并提取特征向量
features1 = []
for keys in tqdm(train_keys):
    path_im = ""
    a = ""
    a = a.join([keys, ".jpg"])
    path_im = os.path.join(path1, a)
    img_t = preprocess(Image.open(path_im).convert('RGB')).unsqueeze(0).to(device)
    model = model.to('cuda')
    feature = model.encode_image(img_t.to('cuda')).cpu().detach().numpy().tolist()
    features1.append(feature)
np.save(feature_file1, features1)
with open(feature_file1, 'rb') as file:
    loaded_data = np.load(file)
print(loaded_data.shape)
print('pass1')
