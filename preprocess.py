import nltk
import os
import pandas as pd
import json
import copy
from tensorflow import keras

os.environ['CUDA_VISIBLE_DEVICES']='-1'
pd.set_option('display.max_columns', 5)

def category_to_vector(category_row):
    categories = ['Monetary', 'Percentage', 'Option', 'Indictor', 'Temporal', 'Quantity', 'Product']

    return list(map(lambda x: 1 if x in category_row else 0, categories))

def category_vectors(data):

    category_data = data[['category']].values[0]
    category_data_vectors = [category_to_vector(category_row[0]) for category_row in category_data]
    print(category_data_vectors)




def main():
    file = 'FinNum_training_rebuilded'
    with open(file + '.json') as f:
        data_training = json.load(f)
    preprocessed_data = []
    for data in data_training:
        for i in range(len(data['category'])):
            data_copy = copy.deepcopy(data)

            text = data_copy['tweet']

            sent_tokenized_text = nltk.sent_tokenize(text)

            tweet = " ".join(filter(lambda x: data["target_num"][i] in x,sent_tokenized_text))

            remove_targets = list(filter(lambda x: x != data['target_num'][i], data['target_num']))
            remove_targets.sort(key=len, reverse=True)
            for target in remove_targets:
                if target == data_copy['target_num'][i]:
                    continue
                if target in data_copy['target_num'][i]:
                    tweet = " ".join(filter(lambda x: x != target, nltk.word_tokenize(tweet)))
                else:
                    tweet = tweet.replace(target, '')

            data_copy['tweet'] = tweet

            data_copy['category'] = data_copy['category'][i]
            data_copy['target_num'] = data_copy['target_num'][i]
            preprocessed_data.append(data_copy)


    data = pd.DataFrame(preprocessed_data)


    data[['category', 'target_num', 'tweet']].to_csv(file + '.csv')



main()