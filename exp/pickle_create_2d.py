import numpy as np
import pickle
import torch
# 图片-resnet50
# with open('./without_GraphAGCE/img_train_2d_64.npy','rb') as f:
#     image_features_train = np.load(f)
# print(image_features_train.shape)
# with open('without_GraphAGCE/img_val_2d_64.npy', 'rb') as f:
#     image_features_val = np.load(f)
# with open('without_GraphAGCE/img_test_2d_64.npy', 'rb') as f:
#     image_features_test = np.load(f)

with open('./clip/image_train1.npy','rb') as f:
    image_features_train = np.load(f)
print(image_features_train.shape)
with open('./clip/image_val1.npy', 'rb') as f:
    image_features_val = np.load(f)
with open('./clip/image_test1.npy', 'rb') as f:
    image_features_test = np.load(f)

# with open('./resnet/img_train_2d_512.npy','rb') as f:
#     image_features_train = np.load(f)
# print(image_features_train.shape)
# with open('./resnet/img_val_2d_512.npy', 'rb') as f:
#     image_features_val = np.load(f)
# with open('./resnet/img_test_2d_512.npy', 'rb') as f:
#     image_features_test = np.load(f)
#
#
# #标签-2元
with open('./MMHS150K/label_train.npy','rb') as f:
    label_train = np.load(f)
print(label_train.shape)
with open('./MMHS150K/label_val.npy','rb') as f:
    label_val = np.load(f)
print(label_val.shape)
with open('./MMHS150K/label_test.npy','rb') as f:
    label_test = np.load(f)
print(label_test.shape)
#1元
# with open('y_train_t.npy','rb') as f:
#     label_train = np.load(f)
# label_train = label_train.astype(int)
# print(label_train.shape)
# with open('y_val_t.npy','rb') as f:
#     label_val = np.load(f)
# label_val = label_val.astype(int)
# print(label_val.shape)
# with open('y_test_t.npy','rb') as f:
#     label_test = np.load(f)
# label_test = label_test.astype(int)
# print(label_test.shape)

#
# # #推文-lstm-未prompt
# with open('./without_GraphAGCE/lstm/tweet_train_2d.npy','rb') as f:
#     docs_tweet_train = np.load(f)
# print(docs_tweet_train.shape)
# with open('./without_GraphAGCE/lstm/tweet_val_2d.npy','rb') as f:
#     docs_tweet_val = np.load(f)
# print(docs_tweet_val.shape)
# with open('./without_GraphAGCE/lstm/tweet_test_2d.npy','rb') as f:
#     docs_tweet_test = np.load(f)
# print(docs_tweet_test.shape)

# # #推文-lstm-gpt-prompt
# with open('./prompt/gpt/tweet_gpt_train_lstm_2d.npy','rb') as f:
#     docs_tweet_train = np.load(f)
# print(docs_tweet_train.shape)
# with open('./prompt/gpt/tweet_gpt_val_lstm_2d.npy','rb') as f:
#     docs_tweet_val = np.load(f)
# print(docs_tweet_val.shape)
# with open('./prompt/gpt/tweet_gpt_test_lstm_2d.npy','rb') as f:
#     docs_tweet_test = np.load(f)
# print(docs_tweet_test.shape)
#
# # #
# # #图片文本-lstm-未prompt
# with open('./without_GraphAGCE/lstm/imgtxt_train_2d.npy','rb') as f:
#     docs_img_train = np.load(f)
# print(docs_img_train.shape)
# with open('./without_GraphAGCE/lstm/imgtxt_val_2d.npy','rb') as f:
#     docs_img_val = np.load(f)
# print(docs_img_val.shape)
# with open('./without_GraphAGCE/lstm/imgtxt_test_2d.npy','rb') as f:
#     docs_img_test = np.load(f)
# print(docs_img_test.shape)

#推文-lstm-prompt
# with open('./prompt/new-llm/tweet_train_lstm_2d.npy','rb') as f:
#     docs_tweet_train = np.load(f)
# print(docs_tweet_train.shape)
# with open('./prompt/new-llm/tweet_val_lstm_2d.npy','rb') as f:
#     docs_tweet_val = np.load(f)
# print(docs_tweet_val.shape)
# with open('./prompt/new-llm/tweet_test_lstm_2d.npy','rb') as f:
#     docs_tweet_test = np.load(f)
# print(docs_tweet_test.shape)
#
# # #图片文本-bert-未prompt
# with open('./without_GraphAGCE/bert/dim=64/train_imgtxt_bert_2d.npy','rb') as f:
#     docs_img_train = np.load(f)
# print(docs_img_train.shape)
# with open('./without_GraphAGCE/bert/dim=64/valid_imgtxt_bert_2d.npy','rb') as f:
#     docs_img_val = np.load(f)
# print(docs_img_val.shape)
# with open('./without_GraphAGCE/bert/dim=64/test_imgtxt_bert_2d.npy','rb') as f:
#     docs_img_test = np.load(f)
# print(docs_img_test.shape)
#
# # #推文-bert-未prompt
# with open('./without_GraphAGCE/bert/dim=64/train_tweet_bert_2d.npy','rb') as f:
#     docs_tweet_train = np.load(f)
# print(docs_tweet_train.shape)
# with open('./without_GraphAGCE/bert/dim=64/valid_tweet_bert_2d.npy','rb') as f:
#     docs_tweet_val = np.load(f)
# print(docs_tweet_val.shape)
# with open('./without_GraphAGCE/bert/dim=64/test_tweet_bert_2d.npy','rb') as f:
#     docs_tweet_test = np.load(f)
# print(docs_tweet_test.shape)

# prompt-flan-t5
with open('./clip/flant5_train.npy','rb') as f:
    prompt_train = np.load(f)
print(prompt_train.shape)
with open('./clip/flant5_val.npy','rb') as f:
    prompt_val = np.load(f)
print(prompt_val.shape)
with open('./clip/flant5_test.npy','rb') as f:
    prompt_test = np.load(f)
print(prompt_test.shape)




# prompt-gpt
# with open('./clip/gpt_train.npy','rb') as f:
#     prompt_train = np.load(f)
# print(prompt_train.shape)
# with open('./clip/gpt_val.npy','rb') as f:
#     prompt_val = np.load(f)
# print(prompt_val.shape)
# with open('./clip/gpt_test.npy','rb') as f:
#     prompt_test = np.load(f)
# print(prompt_test.shape)
#bert
# with open('./bert/gpt_train.npy','rb') as f:
#     prompt_train = np.load(f)
# print(prompt_train.shape)
# with open('./bert/gpt_val.npy','rb') as f:
#     prompt_val = np.load(f)
# print(prompt_val.shape)
# with open('./bert/gpt_test.npy','rb') as f:
#     prompt_test = np.load(f)
# print(prompt_test.shape)
# flant5-clip
# with open('./clip/flant5_train.npy','rb') as f:
#     prompt_train = np.load(f)
# print(prompt_train.shape)
# with open('./clip/flant5_val.npy','rb') as f:
#     prompt_val = np.load(f)
# print(prompt_val.shape)
# with open('./clip/flant5_test.npy','rb') as f:
#     prompt_test = np.load(f)
# print(prompt_test.shape)
#
# with open('./bert/flant5_train.npy','rb') as f:
#     prompt_train = np.load(f)
# print(prompt_train.shape)
# with open('./bert/flant5_val.npy','rb') as f:
#     prompt_val = np.load(f)
# print(prompt_val.shape)
# with open('./bert/flant5_test.npy','rb') as f:
#     prompt_test = np.load(f)
# print(prompt_test.shape)

#tweet-clip
with open('./clip/tweet_train.npy','rb') as f:
    docs_tweet_train = np.load(f)
print(docs_tweet_train.shape)
with open('./clip/tweet_val.npy','rb') as f:
    docs_tweet_val = np.load(f)
print(docs_tweet_val.shape)
with open('./clip/tweet_test.npy','rb') as f:
    docs_tweet_test = np.load(f)
print(docs_tweet_test.shape)
#tweet-bert
# with open('./bert/tweet_train.npy','rb') as f:
#     docs_tweet_train = np.load(f)
# print(docs_tweet_train.shape)
# with open('./bert/tweet_val.npy','rb') as f:
#     docs_tweet_val = np.load(f)
# print(docs_tweet_val.shape)
# with open('./bert/tweet_test.npy','rb') as f:
#     docs_tweet_test = np.load(f)
# print(docs_tweet_test.shape)


#tweet-clip-gpt
# with open('./clip/tweet_gpt_train.npy','rb') as f:
#     docs_tweet_train = np.load(f)
# print(docs_tweet_train.shape)
# with open('./clip/tweet_gpt_val.npy','rb') as f:
#     docs_tweet_val = np.load(f)
# print(docs_tweet_val.shape)
# with open('./clip/tweet_gpt_test.npy','rb') as f:
#     docs_tweet_test = np.load(f)
# print(docs_tweet_test.shape)


# #
# #图片文本-clip
with open('./clip/imgtxt_train.npy','rb') as f:
    docs_img_train = np.load(f)
print(docs_img_train.shape)
with open('./clip/imgtxt_val.npy','rb') as f:
    docs_img_val = np.load(f)
print(docs_img_val.shape)
with open('./clip/imgtxt_test.npy','rb') as f:
    docs_img_test = np.load(f)
print(docs_img_test.shape)
#bert
# with open('./bert/imgtxt_train.npy','rb') as f:
#     docs_img_train = np.load(f)
# print(docs_img_train.shape)
# with open('./bert/imgtxt_val.npy','rb') as f:
#     docs_img_val = np.load(f)
# print(docs_img_val.shape)
# with open('./bert/imgtxt_test.npy','rb') as f:
#     docs_img_test = np.load(f)
# print(docs_img_test.shape)

# #完整训练集
# train = {'docs_tweet': docs_tweet_train, 'docs_img': docs_img_train, 'img': image_features_train, 'label': label_train}
# val = {'docs_tweet': docs_tweet_val, 'docs_img': docs_img_val, 'img': image_features_val, 'label': label_val}
# test = {'docs_tweet': docs_tweet_test, 'docs_img': docs_img_test, 'img': image_features_test, 'label': label_test}
train = {'docs_tweet': docs_tweet_train, 'docs_img': docs_img_train, 'img': image_features_train,'prompt': prompt_train, 'label': label_train}
val = {'docs_tweet': docs_tweet_val, 'docs_img': docs_img_val, 'img': image_features_val,'prompt': prompt_val, 'label': label_val}
test = {'docs_tweet': docs_tweet_test, 'docs_img': docs_img_test, 'img': image_features_test,'prompt': prompt_test, 'label': label_test}
# train = {'docs_tweet': docs_tweet_train, 'docs_img': prompt_train, 'img': docs_img_train, 'label': label_train}
# val = {'docs_tweet': docs_tweet_val, 'docs_img': prompt_val, 'img': docs_img_val, 'label': label_val}
# test = {'docs_tweet': docs_tweet_test, 'docs_img': prompt_test, 'img': docs_img_test, 'label': label_test}
dataset = {'train': train, 'val': val, 'test': test}
#
pic = open('MMHS150K+prompt.pkl','wb')

# #将字典数据存储为一个pkl文件
pickle.dump(dataset,pic)
pic.close()