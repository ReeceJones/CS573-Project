import re
import string
import random
import nltk
import pandas as pd
from typing import List
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams, pad_sequence
from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline
from nltk.lm import MLE
from collections import Counter
from time import time
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
df = pd.read_csv('RAW_recipes.csv').sample(n=100_000)

# Task specification: predict the duration to cook from a given recipe

# select only required columns
df = df[['name', 'id', 'steps']]
print(df.describe())
print(df.head(3))

train_df = df.sample(n=1000)
test_df = df.drop(train_df.index)
test_df.sample(n=200)

sentences = train_df['steps'].tolist()
stop_words = set(stopwords.words('english'))
removal_list = list(stop_words) + list(string.punctuation)

unigrams, bigrams, trigrams = [], [], []


def pre_process(text: str) -> List[str]:
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return word_tokenize(text)


sentences2 = list(map(lambda x: ' '.join(pre_process(x)), sentences))


print('Learning n-grams from dataset of size', len(sentences))
train, vocab = padded_everygram_pipeline(3, sentences2)
lm = MLE(3)
lm.fit(train, vocab)
start = time()
for i, sentence in enumerate(sentences):
    sentence = pre_process(sentence)
    unigrams.extend(list(ngrams(sentence, 1, pad_left=True)))
    bigrams.extend(list(ngrams(sentence, 2, pad_left=True)))
    trigrams.extend((list(ngrams(sentence, 3, pad_left=True))))
    if i % int(len(sentences)/10) == 0:
        print('Processed', i, 'sentences. (Last', int(len(sentences)/10), 'in', time()-start, 'seconds)')
        start = time()


unigram_counter, bigram_counter, trigram_counter = Counter(unigrams), Counter(bigrams), Counter(trigrams)
unigram_total, bigram_total, trigram_total = sum(unigram_counter.values()), sum(bigram_counter.values()), sum(trigram_counter.values())


def get_next_token(input_str):
    tokens = pre_process(input_str)
    last = tokens[-1]
    second_last = tokens[-2] if len(tokens) >= 2 else None
    third_last = tokens[-3] if len(tokens) >= 3 else None
    choices = list(filter(lambda x: x[0] == second_last and x[1] == last, trigram_counter.elements()))
    if not choices:
        choices = list(filter(lambda x: x[0] == last, bigram_counter.elements()))
    if not choices:
        choices = list(unigram_counter.elements())

    return random.choice(choices)[-1]


def evaluate():
    global test_df
    unigram_pp, bigram_pp, trigram_pp = 1, 1, 1
    for _, row in test_df.iterrows():
        temp = 1
        tokens = pre_process(row['steps'])
        for token in tokens:
            temp *= unigram_total/unigram_counter[(token,)]
        test_df['unigram_pp'] = temp


if __name__ == '__main__':

    evaluate()
    print(test_df.describe())

    print('Type some text and hit enter. Type q to quit')
    working_text = ''
    while True:
        user_input = input('User input: ')
        if len(user_input) <= 0:
            continue
        if user_input == 'q':
            break
        working_text += ' ' + user_input
        next_token = get_next_token(working_text)
        print('Prediction:', working_text, next_token)



