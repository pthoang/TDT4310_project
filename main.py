import nltk
import os
import pandas as pd
import json
import numpy as np
from tensorflow import keras
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVR
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

os.environ['CUDA_VISIBLE_DEVICES']='-1'

class_mapping = {
    'Monetary': [1, 0, 0, 0, 0, 0, 0],
    'Percentage': [0, 1, 0, 0, 0, 0, 0],
    'Option': [0, 0, 1, 0, 0, 0, 0],
    'Indicator': [0, 0, 0, 1, 0, 0, 0],
    'Temporal': [0, 0, 0, 0, 1, 0, 0],
    'Quantity': [0, 0, 0, 0, 0, 1, 0],
    'Product Number': [0, 0, 0, 0, 0, 0, 1],
}

def build_model(n_features, max_length, n_classes, batch_size):
    lstm = keras.Sequential()
    lstm.add(keras.layers.Embedding(n_features + 1, 128, input_length=max_length,
                                    ))
    lstm.add(keras.layers.Dropout(0.4))
    lstm.add(keras.layers.LSTM(units=256, recurrent_dropout=0.2, dropout=0.2))
    lstm.add(keras.layers.Dropout(0.2))
    lstm.add(keras.layers.Dense(n_classes, activation='sigmoid'))
    lstm.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.Adam(0.001),
        metrics=['accuracy']
    )
    return lstm

def run_lstm(all_data):
    tweet_tokenizer = TweetTokenizer(reduce_len=True)

    category_index = 0
    tweet_index = 2
    classified_tweets = []
    max_tweet_length = 0
    for row in all_data.values:
        tokenized_tweet = tweet_tokenizer.tokenize(row[tweet_index])
        max_tweet_length = max_tweet_length if max_tweet_length > len(tokenized_tweet) else len(tokenized_tweet)
        classified_tweets.append((row[tweet_index], class_mapping[row[category_index]]))

    keras_tokenizer = keras.preprocessing.text.Tokenizer(filters='')
    padder = keras.preprocessing.sequence.pad_sequences

    keras_tokenizer.fit_on_texts([tweet[0] for tweet in classified_tweets])

    encoded_docs = keras_tokenizer.texts_to_sequences([tweet[0] for tweet in classified_tweets])
    max_length = np.max([len(x) for x in encoded_docs])

    padded_encoded_docs = padder(encoded_docs, maxlen=max_length, padding='post')

    X_train, X_test, y_train, y_test = train_test_split(padded_encoded_docs,
                                                        np.array([tweet[1]
                                                                  for tweet in classified_tweets]),
                                                        test_size=0.1)

    batch_size = len(X_train)
    model = build_model(len(keras_tokenizer.word_index), max_length, len(class_mapping), batch_size)

    model.fit(X_train,
              y_train, epochs=5, batch_size=batch_size)

    results = model.predict(X_test)

    micro = f1_score([np.argmax(label) for label in y_test],
                                            [np.argmax(label) for label in results],
                                            average='micro')
    macro = f1_score([np.argmax(label) for label in y_test],
                                            [np.argmax(label) for label in results],
                                            average='macro')

    print('F-score(micro): ' + str(micro))
    print('F-score(macro): ' + str(macro))

    return micro, macro

def run_other(all_data, model='svc'):

    models = {
        'svc': LinearSVC(),
        'mnb': MultinomialNB(),
        'log_reg': LogisticRegression(),
        'gbc': GradientBoostingClassifier()
    }

    tfidf = TfidfVectorizer(sublinear_tf=False, min_df=2, norm='l2', ngram_range=(1,2), stop_words=None)

    features = tfidf.fit_transform(all_data['tweet'])
    labels = [class_mapping[category].index(1) for category in all_data['category'].values]

    model = LinearSVC()

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1)

    model.fit(X_train, y_train)

    results = model.predict(X_test)

    micro = f1_score(y_test, results, average='micro')
    macro = f1_score(y_test, results, average='macro')

    print('F-score(micro): ' + str(micro))
    print('F-score(macro): ' + str(macro))

    return micro, macro



def main():
    data_1 = pd.read_csv('FinNum_training_rebuilded.csv')
    data_2 = pd.read_csv('FinNum_dev_rebuilded.csv')

    data_1 = data_1[['category', 'target_num', 'tweet']]
    data_2 = data_2[['category', 'target_num', 'tweet']]

    all_data = pd.concat([data_1, data_2]).reset_index().drop(columns=['index'])

    total_micro = 0
    total_macro = 0
    runs = 100
    for i in range(runs):

        # micro, macro = run_lstm(all_data)

        micro, macro = run_other(all_data, model='gbc')

        total_micro += micro
        total_macro += macro

    print('Avg. F-score(micro): ' + str(total_micro/runs))
    print('Avg. F-score(macro): ' + str(total_macro/runs))

main()