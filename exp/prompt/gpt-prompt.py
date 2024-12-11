import openai
import numpy as np
from tqdm import tqdm
openai.api_base = ''
openai.api_key = ''

train_tweet = np.load('./new_data/tweet_train_text_unprompt.npy')
val_tweet = np.load('./new_data/tweet_val_text_unprompt.npy')
test_tweet = np.load('./new_data/tweet_test_text_unprompt.npy')

if __name__ == '__main__':

  val_list = []
  for row in tqdm(val_tweet, desc="Processing tweets"):
    # print("row:" + row)
    new_context1 = f'Determine the implied meaning the following text contains to its target demographic group.\nPost: "{row}"\nAnswer: Please begin with "this text contains" in the sentence, and summary in less than ten words. '
    prompt1 = new_context1
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system",
         "content": "You are a helpful assistant designed to distinguish between hate speech and non-hate speech."},
        {"role": "user",
         "content": prompt1},
      ]
    )
    output1 = response['choices'][0]['message']['content']
    # print("output1:" + output1)
    implied_text = f" {output1}"
    val_list.append(implied_text)
  val_list = np.array(val_list)
  np.save('text_val_implied_gpt.npy', val_list)

  test_list = []
  for row in tqdm(test_tweet, desc="Processing tweets"):
      # print("row:" + row)
      new_context1 = f'Determine the implied meaning the following text contains to its target demographic group.\nPost: "{row}"\nAnswer: Please begin with "this text contains" in the sentence, and summary in less than ten words. '
      prompt1 = new_context1
      response = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          messages=[
              {"role": "system",
               "content": "You are a helpful assistant designed to distinguish between hate speech and non-hate speech."},
              {"role": "user",
               "content": prompt1},
          ]
      )
      output1 = response['choices'][0]['message']['content']
      # print("output1:" + output1)
      implied_text = f" {output1}"
      test_list.append(implied_text)
  test_list = np.array(test_list)
  np.save('text_test_implied_gpt.npy', test_list)

    batch_size = 10000
    num_batches = 13
    for batch_idx in range(2, num_batches + 1):
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx + 1) * batch_size

        batch_tweets = train_tweet[start_idx:end_idx]
        train_list = []
        for row in tqdm(batch_tweets, desc="Processing tweets"):
            # print("row:" + row)
            new_context1 = f'Determine the implied meaning the following text contains to its target demographic group.\nPost: "{row}"\nAnswer: Please begin with "this text contains" in the sentence, and summary in less than ten words. '
            prompt1 = new_context1
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system",
                     "content": "You are a helpful assistant designed to distinguish between hate speech and non-hate speech."},
                    {"role": "user",
                     "content": prompt1},
                ]
            )
            output1 = response['choices'][0]['message']['content']
            # print("output1:" + output1)
            implied_text = f" {output1}"
            train_list.append(implied_text)
        train_list = np.array(train_list)
        np.save(f'text_train_implied_gpt_{batch_idx + 1}.npy', train_list)