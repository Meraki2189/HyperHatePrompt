import numpy as np

def clean(data1):
    filtered_data = [line.replace("This text contains ", "") for line in data1]
    filtered_data = [line.replace("This text ", "") for line in filtered_data]
    filtered_data = [line.replace("Summary:", "") for line in filtered_data]
    filtered_data = [line.replace(" implies that", "") for line in filtered_data]
    filtered_data = [line.replace(" target demographic group", "") for line in filtered_data]
    filtered_data = [line.replace(" specific demographic group", "") for line in filtered_data]
    filtered_data = [line.replace(" specific demographic", "") for line in filtered_data]
    filtered_data = [line.replace(" demographic group", "") for line in filtered_data]
    filtered_data = [line.replace(" mentioned in", "") for line in filtered_data]
    filtered_data = [line.replace(".", "") for line in filtered_data]
    filtered_data = [line.lower() for line in filtered_data]
    filtered_data_array = np.array(filtered_data)
    return filtered_data_array

with open('./gpt/text_val_implied_gpt.npy','rb') as f:
    data1 = np.load(f)
clean_data1 = clean(data1)
np.save('./gpt/text_val_implied_gpt_new.npy',clean_data1)
# # 将数据保存为.txt文件
# np.savetxt('./gpt/val_gpt.txt', data1, fmt='%s', encoding='utf-8')

with open('./gpt/text_test_implied_gpt.npy','rb') as f:
    data2 = np.load(f)
clean_data2 = clean(data2)
np.save('./gpt/text_test_implied_gpt_new.npy',clean_data2)
# # 将数据保存为.txt文件
# np.savetxt('./gpt/test_gpt.txt', data2, fmt='%s', encoding='utf-8')

with open('./gpt/text_train_implied_gpt.npy','rb') as f:
    data3 = np.load(f)
clean_data3 = clean(data3)
np.save('./gpt/text_train_implied_gpt_new.npy',clean_data3)
# 将数据保存为.txt文件
np.savetxt('./gpt/train_gpt.txt', data3, fmt='%s', encoding='utf-8')


with open('./new-llm/tweet_train_implied.npy','rb') as f:
    data1 = np.load(f)

# # 删除每行文本中的 "the tweet" 部分
filtered_data = [line.replace(" the tweet", "") for line in data1]
filtered_data = [line.replace(" implies that", "") for line in filtered_data]
filtered_data = [line.replace(" The tweet", "") for line in filtered_data]
filtered_data = [line.replace(" target demographic group", "") for line in filtered_data]
filtered_data = [line.replace(" mentioned in", "") for line in filtered_data]
filtered_data = [line.replace(".", "") for line in filtered_data]
filtered_data = [line.lower() for line in filtered_data]
# # # 转换为 NumPy 数组
filtered_data_array = np.array(filtered_data)
np.save('./new-llm/tweet_train_implied_new.npy', filtered_data_array)
#
with open('./new-llm/tweet_val_implied.npy','rb') as f:
    data1 = np.load(f)

# 删除每行文本中的 "the tweet" 部分
filtered_data = [line.replace(" the tweet", "") for line in data1]
filtered_data = [line.replace(" implies that", "") for line in filtered_data]
filtered_data = [line.replace(" The tweet", "") for line in filtered_data]
filtered_data = [line.replace(" target demographic group", "") for line in filtered_data]
filtered_data = [line.replace(" mentioned in", "") for line in filtered_data]
filtered_data = [line.replace(".", "") for line in filtered_data]
filtered_data = [line.lower() for line in filtered_data]
# # 转换为 NumPy 数组
filtered_data_array = np.array(filtered_data)
np.save('./new-llm/tweet_val_implied_new.npy', filtered_data_array)

with open('./new-llm/tweet_test_implied.npy','rb') as f:
    data1 = np.load(f)

# 删除每行文本中的 "the tweet" 部分
filtered_data = [line.replace(" the tweet", "") for line in data1]
filtered_data = [line.replace(" implies that", "") for line in filtered_data]
filtered_data = [line.replace(" The tweet", "") for line in filtered_data]
filtered_data = [line.replace(" target demographic group", "") for line in filtered_data]
filtered_data = [line.replace(" mentioned in", "") for line in filtered_data]
filtered_data = [line.replace(".", "") for line in filtered_data]
filtered_data = [line.lower() for line in filtered_data]
# # 转换为 NumPy 数组
filtered_data_array = np.array(filtered_data)
np.save('./new-llm/tweet_test_implied_new.npy', filtered_data_array)


with open('./new_data/tweet_train_text_unprompt.npy','rb') as f:
    data1 = np.load(f)
with open('./gpt/text_train_implied_gpt_new.npy','rb') as f:
    data2 = np.load(f)
data = np.array([str(item1) + str(item2) for item1, item2 in zip(data1, data2)])
np.save('./gpt/tweet_train_text_prompt.npy',data)

with open('./new_data/tweet_val_text_unprompt.npy','rb') as f:
    data1 = np.load(f)
with open('./gpt/text_val_implied_gpt_new.npy','rb') as f:
    data2 = np.load(f)
data = np.array([str(item1) + str(item2) for item1, item2 in zip(data1, data2)])
np.save('./gpt/tweet_val_text_prompt.npy',data)

with open('./new_data/tweet_test_text_unprompt.npy','rb') as f:
    data1 = np.load(f)
with open('./gpt/text_test_implied_gpt_new.npy','rb') as f:
    data2 = np.load(f)
data = np.array([str(item1) + str(item2) for item1, item2 in zip(data1, data2)])
np.save('./gpt/tweet_test_text_prompt.npy',data)