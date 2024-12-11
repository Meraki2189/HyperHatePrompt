from transformers import T5Tokenizer, T5ForConditionalGeneration
import numpy as np
from tqdm import tqdm
import re
train_tweet = np.load('tweet_train_text_unprompt.npy')
val_tweet = np.load('tweet_val_text_unprompt.npy')
test_tweet = np.load('tweet_test_text_unprompt.npy')


tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", device_map="auto")

# val_list = []
# target_list = []
implied_list = []
for row in tqdm(val_tweet, desc="Processing tweets"):
    # print("row:" + row)
    new_context1 = f'Given the tweet "{row}", '
    prompt1 = new_context1 + f'  What is the target demographic group of the tweet?'
    # example = " For example, given the tweet 'It is incredible what we did in these years. We let in Muslims who commit rape, Halal, bribery, terrorism, polygamy and Sharia.', we can include: the target demographic group of the tweet is Muslims."
    # prompt1 = prompt1 + example
    input_ids = tokenizer(prompt1, return_tensors="pt").input_ids.to("cuda")
    output1 = model.generate(input_ids)
    target_group = tokenizer.decode(output1[0])
    target_group = target_group.replace("<pad> ", "").replace("</s>", "")
    # print("output1:" + target_group)

    new_context2 = f'Given the tweet "{row}", the target demographic group mentioned in the tweet is  "{target_group}".'
    prompt2 = new_context2 + f' Articulate what the implied stand of point the tweet holds towards "{target_group}". And why?'
    # example = " For example, given the tweet 'It is incredible what we did in these years. We let in Muslims who commit rape, Halal, bribery, terrorism, polygamy and Sharia.', and the target demographic group of the tweet is Muslims, we can include: The implied perspective is based on misinformation, bias, and a lack of understanding about the diversity within the Muslim community. It wrongly assumes that all Muslims share the same negative behaviors and beliefs, which is an unfair and inaccurate characterization."
    # prompt2 = prompt2 + example
    input_ids2 = tokenizer(prompt2, return_tensors="pt").input_ids.to("cuda")
    output2 = model.generate(input_ids2, max_length = 50)
    implied_meaning = tokenizer.decode(output2[0])
    implied_meaning = implied_meaning.replace("<pad> ", "").replace("</s>", "")
    # print("output2:" + implied_meaning)
    # combined_text = f"{row} {implied_meaning}"
    # val_list.append(combined_text)
    # target_text = f"{target_group}"
    implied_text = f" {implied_meaning}"
    # target_list.append(target_text)
    implied_list.append(implied_text)
# val_list = np.array(val_list)
# np.save('tweet_val_prompt.npy',val_list)
# target_list = np.array(target_list)
# np.save('tweet_val_target.npy',target_list)
implied_list = np.array(implied_list)
np.save('tweet_val_implied.npy',implied_list)
#
# test_list = []
# target_list = []
implied_list = []
for row in tqdm(test_tweet, desc="Processing tweets"):
    # print("row:" + row)
    new_context1 = f'Given the tweet "{row}", '
    prompt1 = new_context1 + f' What is the target demographic group of the tweet?'
    input_ids = tokenizer(prompt1, return_tensors="pt").input_ids.to("cuda")
    output1 = model.generate(input_ids)
    target_group = tokenizer.decode(output1[0])
    target_group = target_group.replace("<pad> ", "").replace("</s>", "")
    # print("output1:" + target_group)
    new_context2 = f'Given the tweet "{row}", the target demographic group mentioned in the tweet is  "{target_group}".'
    prompt2 = new_context2 + f' Articulate what the implied stand of point the tweet holds towards "{target_group}". And why?'
    input_ids2 = tokenizer(prompt2, return_tensors="pt").input_ids.to("cuda")
    output2 = model.generate(input_ids2, max_length = 50)
    implied_meaning = tokenizer.decode(output2[0])
    implied_meaning = implied_meaning.replace("<pad> ", "").replace("</s>", "")
    # print("output2:" + implied_meaning)
    # combined_text = f"{row} {implied_meaning}"
    # test_list.append(combined_text)
    # target_text = f"{target_group}"
    implied_text = f" {implied_meaning}"
    # target_list.append(target_text)
    implied_list.append(implied_text)
# test_list = np.array(test_list)
# np.save('tweet_test_prompt.npy', test_list)
# target_list = np.array(target_list)
# np.save('tweet_test_target.npy', target_list)
implied_list = np.array(implied_list)
np.save('tweet_test_implied.npy', implied_list)

# 第一个prompt应生成target group，第二个prompt应生成implied meaning
# train_list = []
# target_list = []
implied_list = []
for row in tqdm(train_tweet, desc="Processing tweets"):
    # print("row:" + row)
    # 清理文本，去除非 ASCII 字符
    new_context1 = f'Given the tweet "{row}", '
    prompt1 = new_context1 + f' What is the target demographic group of the tweet?'
    input_ids = tokenizer(prompt1, return_tensors="pt").input_ids.to("cuda")
    output1 = model.generate(input_ids)
    target_group = tokenizer.decode(output1[0])
    target_group = target_group.replace("<pad> ", "").replace("</s>", "")
    # print("output1:" + target_group)
    new_context2 = f'Given the tweet "{row}", the target demographic group mentioned in the tweet is  "{target_group}".'
    prompt2 = new_context2 + f' Articulate what the implied stand of point the tweet holds towards "{target_group}". And why?'
    input_ids2 = tokenizer(prompt2, return_tensors="pt").input_ids.to("cuda")
    output2 = model.generate(input_ids2, max_length = 50)
    implied_meaning = tokenizer.decode(output2[0])
    implied_meaning = implied_meaning.replace("<pad> ", "").replace("</s>", "")
    # print("output2:" + implied_meaning)
    # combined_text = f"{row} {implied_meaning}"
    # train_list.append(combined_text)
    # target_text = f"{target_group}"
    implied_text = f" {implied_meaning}"
    # target_list.append(target_text)
    implied_list.append(implied_text)
# train_list = np.array(train_list)
# np.save('tweet_train_prompt.npy', train_list)
# target_list = np.array(target_list)
# np.save('tweet_train_target.npy', target_list)
implied_list = np.array(implied_list)
np.save('tweet_train_implied.npy', implied_list)

