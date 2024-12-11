import torch
import torch.nn as nn
import os
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from torch.autograd import Variable

resnet = models.resnet50(pretrained=True)
resnet.fc = nn.Linear(2048, 512)
# resnet.fc = torch.nn.Identity()
resnet = resnet.to('cuda:0')
resnet.eval()
def get_image_features(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).to('cuda:0')
    with torch.no_grad():
        encoding = resnet(input_tensor.unsqueeze(0))
    return encoding
    # # 切片参数
    # # block_size = 32
    # block_size = 32
    # input_tensor = transform(image).to('cuda:0')
    # # print(input_tensor.shape)
    # # 获取图像大小
    # image_width, image_height = 224, 224
    # # 计算块的数量 长宽均为7个
    # num_blocks_horizontal = image_width // block_size
    # num_blocks_vertical = image_height // block_size
    # num_blocks = num_blocks_horizontal * num_blocks_vertical
    # # 创建编码向量
    # # encoding = torch.empty((num_blocks, 2048))
    # encoding_blocks = torch.zeros(num_blocks, 128).to('cuda:0')  # 存储编码结果的数组，尺寸为[块数, 编码维度]
    #
    # for i in range(num_blocks_vertical):
    #     for j in range(num_blocks_horizontal):
    #         # block_image = image.crop((j * block_size, i * block_size, (j + 1) * block_size, (i + 1) * block_size))
    #         block_image = input_tensor[:, i * block_size: (i + 1) * block_size, j * block_size: (j + 1) * block_size]
    #         # block_input_tensor = transform(block_image)
    #         with torch.no_grad():
    #             block_encoding = resnet(block_image.unsqueeze(0))
    #         encoding_blocks[i * num_blocks_horizontal + j] = torch.squeeze(block_encoding)
    # return encoding_blocks

feature_file2 = './without_GraphAGCE/img_val_2d_512.npy'
val_keys = open('./dataset1/splits/val_ids.txt')
val_keys = val_keys.read()
val_keys = val_keys.splitlines()
# image_val = np.zeros((5000, 256, 256, 3))  # 图片
# image_val = np.zeros((32, 32, 3))  # 图片
# zeros = np.zeros((32, 32))
i = 0
path1 = Path('./dataset1/img_resized')  # 图片
# 遍历文件夹中的所有图像文件，并提取特征向量
features2 = []
for keys in tqdm(val_keys):
    path_im = ""
    a = ""
    a = a.join([keys, ".jpg"])
    path_im = os.path.join(path1, a)
    img_t = Image.open(path_im).convert('RGB')
    feature = get_image_features(img_t).cpu().numpy().tolist()
    features2.append(feature)
np.save(feature_file2, features2)
with open(feature_file2, 'rb') as file:
    loaded_data = np.load(file)
print(loaded_data.shape)
print('pass2')
#
#
feature_file3 = './without_GraphAGCE/img_test_2d_512.npy'
test_keys = open('./dataset1/splits/test_ids.txt')
test_keys = test_keys.read()
test_keys = test_keys.splitlines()
# image_test = np.zeros((32, 32, 3))  # 图片
# zeros = np.zeros((32, 32))
i = 0
path1 = Path('./dataset1/img_resized')  # 图片
# 遍历文件夹中的所有图像文件，并提取特征向量
features3 = []
for keys in tqdm(test_keys):
    path_im = ""
    a = ""
    a = a.join([keys, ".jpg"])
    path_im = os.path.join(path1, a)
    img_t = Image.open(path_im).convert('RGB')
    # if img_t.mode != 'RGB':
    #     null = img_t.resize((32, 32))
    #     image_test = np.stack([null, zeros, zeros], axis=-1)
    # else:
    #     image_test = img_t.resize((32, 32)) # 图片
    feature = get_image_features(img_t).cpu().numpy().tolist()
    features3.append(feature)
# 将特征向量保存到.npy文件中
np.save(feature_file3, features3)
with open(feature_file3, 'rb') as file:
    loaded_data = np.load(file)
print(loaded_data.shape)
print('pass3')


# 定义特征向量保存文件名和文件夹路径
feature_file1 = './without_GraphAGCE/img_train_2d_512.npy'
train_keys = open('./dataset1/splits/train_ids.txt')
train_keys = train_keys.read()
train_keys = train_keys.splitlines()
# image_train = np.zeros((32, 32, 3))  # 图片
# zeros = np.zeros((32, 32))
i = 0
path1 = Path('./dataset1/img_resized')  # 图片
# 遍历文件夹中的所有图像文件，并提取特征向量
features1 = []
for keys in tqdm(train_keys):
    path_im = ""
    a = ""
    a = a.join([keys, ".jpg"])
    path_im = os.path.join(path1, a)
    img_t = Image.open(path_im).convert('RGB')
    feature = get_image_features(img_t).cpu().numpy().tolist()
    features1.append(feature)
# 将特征向量保存到.npy文件中
np.save(feature_file1, features1)
with open(feature_file1, 'rb') as file:
    loaded_data = np.load(file)
print(loaded_data.shape)
# # 创建一个空的.npy文件或者确保.npy文件已经存在
# batch_size = 10000 # 每批次处理的图像数量
# total_samples = 134823 # 总共的样本数
# total_batches = total_samples // batch_size + 1
# # 遍历文件夹中的所有图像文件，并提取特征向量
# # features1 = []
# for batch_idx in range(total_batches):
#     feature_file1 = 'image_features_train_49_128_' + str(batch_idx) + '.npy'
#     start_idx = batch_idx * batch_size
#     end_idx = min((batch_idx + 1) * batch_size, total_samples)
#     batch_keys = train_keys[start_idx:end_idx]
#     batch_features = [] # 存储当前批次的特征
#     for keys in tqdm(batch_keys):
#         path_im = ""
#         a = ""
#         a = a.join([keys, ".jpg"])
#         path_im = os.path.join(path1, a)
#         img_t = Image.open(path_im).convert('RGB')
#         # if img_t.mode != 'RGB':
#         #     null = img_t.resize((32, 32))
#         #     image_train = np.stack([null, zeros, zeros], axis=-1)
#         # else:
#         #     image_train = img_t.resize((32, 32)) # 图片
#         feature = get_image_features(img_t).cpu().numpy().tolist()
#         batch_features.append(feature)
#         del feature
# # existing_features = np.load(feature_file1, allow_pickle=True)
# # print(existing_features)
# # print(batch_features)
# # updated_features = np.concatenate((existing_features, batch_features), axis=0)
# # np.save(feature_file1, updated_features)
# # del existing_features
# # del
#     np.save(feature_file1, batch_features)
#     del batch_features
#     with open(feature_file1, 'rb') as file:
#         loaded_data = np.load(file)
#         print(loaded_data.shape)
#         # features1.append(feature)
#         # print(np.array(feature).shape)

# np.save(feature_file1, features1)
# with open(feature_file1, 'rb') as file:
# loaded_data = np.load(file)
# print(loaded_data.shape)
print('pass1')

# feature_file = 'image_features.npy'
# directory = 'image_folder'
# path1 = Path('./test_image')  # 图片
# # 遍历文件夹中的所有图像文件，并提取特征向量
# features = []
# path_im = ""
# path_im = os.path.join(path1, 'v2-3bf5df9ced6b54f4ca4b1a01dbeeadb5_r.jpg')
# img_t = Image.open(path_im).convert('RGB')
#     # if img_t.mode != 'RGB':
#     #     null = img_t.resize((32, 32))
#     #     image_train[i] = np.stack([null, zeros, zeros], axis=-1)
#     # else:
#     #     image_train[i] = img_t.resize((32, 32)) # 图片
# feature = get_image_features(img_t)
# features.append(feature)
# print(feature.shape)