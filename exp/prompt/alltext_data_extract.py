import re  # Regex package
import os
from pathlib import Path
from os import path
import json
import numpy as np
import nltk
import emoji
nltk.download('punkt')
# Function for cleaning tweet text
def convert_to_original(phrase):
  # specific
  phrase = re.sub(r"won\'t", "will not", phrase)
  phrase = re.sub(r"can\'t", "can not", phrase)
  # general
  phrase = re.sub(r"n\'t", " not", phrase)
  phrase = re.sub(r"\'re", " are", phrase)
  phrase = re.sub(r"\'s", " is", phrase)
  phrase = re.sub(r"\'d", " would", phrase)
  phrase = re.sub(r"\'ll", " will", phrase)
  phrase = re.sub(r"\'t", " not", phrase)
  phrase = re.sub(r"\'ve", " have", phrase)
  phrase = re.sub(r"\'m", " am", phrase)
  return phrase


def clean_symbols(text):
  '''Cleans tweet text by homogenizing mentions (@user), hashtags (#hashtag),
  URL (https/), removing double or more spaces and transforms to lower case.'''
  # return ' '.join(re.sub('\w+:\/\/\S+', 'https/',
  #               re.sub('#[A-Za-z0-9]+', '#hashtag',
  #                      re.sub('@[A-Za-z0-9]+', '@user', astring))).lower().split())
  # text data cleaning
  # lower casing
  text = str(text).lower()

  # remove unicode strings
  text = re.sub(r'(\\u[0-9A-Fa-f]+)', r'', text)

  # remove non-ascii characters
  text = re.sub(r'[^\x00-\x7f]', r'', text)

  # remove @user indicaters
  text = re.sub(r'@\w+', r'', text)

  # remove digits
  text = re.sub(r'\d', r'', text)

  # remove '#' symbols inside strings
  text = re.sub(r'#', r'', text)

  # remove rt tag in html
  text = re.sub(r'rt', r'', text)

  # remove “ampersand” reference in html
  text = re.sub(r'amp|&amp', r'', text)

  # remove urls
  text = re.sub(r'http\S+', r'', text)

  # remove multiple white spaces
  text = re.sub(r'[\s]+', r' ', text)

  # remove multiple break-line with single white space.
  text = re.sub(r'[\n]+', r' ', text)

  # split the text (tokenization)
  text = nltk.word_tokenize(text)

  # convert abbreviated form to original form
  text = convert_to_original(' '.join(text)).split()

  # remove short strings with length=1 and length=2
  text = [word for word in text if not len(word) in [1, 2]]

  # remove non-alphanumeric characters
  text = [word for word in text if word.isalpha()]

  return ' '.join(text)


# Creating data and label dataset
def for_text_label(adict, subset=False):
    '''Extracts from dictionary adict, two lists:
    text and labels. Subset option is for managing only 10
    observations.'''
    count = 0
    text = []
    labels3 = []

    for i, j in adict.items():
        temp_text = clean_symbols(adict[i]['tweet_text'])
        text.append(temp_text)

        temp_labels = adict[i]['labels']
        labels3.append(temp_labels)

        if subset == True:
            count += 1
        if count == 5:
            break
    return text, labels3


def for_imgtext_label(adict, subset=False):
    '''Extracts from dictionary adict, two lists:
    text and labels. Subset option is for managing only 10
    observations.'''
    count = 0
    text = []
    labels3 = []
    path2 = Path('./dataset1/img_txt')  # 图片文字
    for i, j in adict.items():
        b = ""
        b = b.join([i, ".json"])
        path_im_txt = os.path.join(path2, b)
        if path.exists(path_im_txt):
            img_text = json.load(open(path_im_txt))
            imgtxt = clean_symbols(img_text['img_text'])
        else:
            imgtxt = ' ' #若没有该数据

        text.append(imgtxt)

        temp_labels = adict[i]['labels']
        labels3.append(temp_labels)

        if subset == True:
            count += 1
        if count == 5:
            break
    return text, labels3

def for_bothtext_label(adict, subset=False):
    '''Extracts from dictionary adict, two lists:
    text and labels. Subset option is for managing only 10
    observations.'''
    count = 0
    text = []
    labels3 = []
    path2 = Path('./dataset1/img_txt')  # 图片文字
    for i, j in adict.items():
        tweet_text = clean_symbols(adict[i]['tweet_text'])
        b = ""
        b = b.join([i, ".json"])
        path_im_txt = os.path.join(path2, b)
        if path.exists(path_im_txt):
            img_text = json.load(open(path_im_txt))
            imgtxt = clean_symbols(img_text['img_text'])
        else:
            imgtxt = ' ' #若没有该数据
        bothtxt = tweet_text + ' ' + imgtxt
        text.append(bothtxt)

        if subset == True:
            count += 1
        if count == 5:
            break
    return text

# Majority vote function
def majority_vote(alist):
    '''Using majority vote, classifies whether it is
    hate speech (= 1) or not (= 0).'''
    label_res = []

    for i in alist:
        zero_count = 0

        for tag in i:
            if tag == 0:
                zero_count += 1

        if zero_count >= 2:
            hate = 0
        else:
            hate = 1

        label_res.append(hate)

    return label_res


# Annotators' agreement measurement

def annot_agreement(alist):
    '''Measures agreement based on a list with three votes. Returns the sum of
    all zeros, one zero, two zeros, all hate labels, and equal labelling of
    hate (given that all annotators label as hate) per tweet.'''
    all_zeros = 0
    one_zero = 0
    two_zero = 0
    all_hate = 0
    # equal = 0

    for i in alist:
        zero_count = 0
        for tag in i:
            if tag == 0:
                zero_count += 1

        if zero_count == 3:
            all_zeros += 1
        elif zero_count == 2:
            two_zero += 1
        elif zero_count == 1:
            one_zero += 1
        else:
            all_hate += 1


    return all_zeros, one_zero, two_zero, all_hate


# Load MMHS150K dataset
with open('./dataset1/MMHS150K_GT.json') as f:
    data = json.load(f)
# Load training, validation and test IDs
id_train0 = open('./dataset1/splits/train_ids.txt').read()
id_train = id_train0.split()
id_val0 = open('./dataset1/splits/val_ids.txt').read()
id_val = id_val0.split()
id_test0 = open('./dataset1/splits/test_ids.txt').read()
id_test = id_test0.split()

# Create a dataset (list of tweet_text and labels) for each set: training,␣,→ validation and test
dict_train = {x: data[x] for x in id_train}
dict_val = {x: data[x] for x in id_val}
dict_test = {x: data[x] for x in id_test}

# Creating data and label dataset
tweet_train, labels3_train = for_text_label(dict_train)
tweet_val, labels3_val = for_text_label(dict_val)
tweet_test, labels3_test = for_text_label(dict_test)
tweet_test = np.array(tweet_test)
tweet_val = np.array(tweet_val)
tweet_train = np.array(tweet_train)

imgtxt_train, _ = for_imgtext_label(dict_train)
imgtxt_val, _ = for_imgtext_label(dict_val)
imgtxt_test, _ = for_imgtext_label(dict_test)
imgtxt_test = np.array(imgtxt_test)
imgtxt_val = np.array(imgtxt_val)
imgtxt_train = np.array(imgtxt_train)

bothtext_train = for_bothtext_label(dict_train)
bothtext_val = for_bothtext_label(dict_val)
bothtext_test = for_bothtext_label(dict_test)


np.save('tweet_train_text_unprompt.npy', tweet_train)
np.save('tweet_val_text_unprompt.npy', tweet_val)
np.save('tweet_test_text_unprompt.npy', tweet_test)

np.save('imgtxt_train_text_unprompt.npy', imgtxt_train)
np.save('imgtxt_val_text_unprompt.npy', imgtxt_val)
np.save('imgtxt_test_text_unprompt.npy', imgtxt_test)

np.save('bothtext_train_text_unprompt.npy', bothtext_train)
np.save('bothtext_val_text_unprompt.npy', bothtext_val)
np.save('bothtext_test_text_unprompt.npy', bothtext_test)