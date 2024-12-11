import re # Regex package
import os
from pathlib import Path
from os import path
import json
import nltk
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
def for_text_label(adict, subset = False):
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

def for_imgtext_label(adict, subset = False):
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
      imgtxt = ' '

    text.append(imgtxt)

    temp_labels = adict[i]['labels']
    labels3.append(temp_labels)

    if subset == True:
      count += 1
    if count == 5:
      break
  return text, labels3



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
    #equal = 0

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

        # Count agreement
        #if i[0] == i[1] == i[2]:
          #equal += 1

    return all_zeros, one_zero, two_zero, all_hate