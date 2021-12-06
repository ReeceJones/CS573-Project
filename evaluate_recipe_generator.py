import re
from typing import List
import numpy as np
import pandas as pd
from nltk.lm import MLE
from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline
from nltk.tokenize import word_tokenize
from nltk.util import bigrams, ngrams
from time import time
import winsound

winsound.Beep(2500, 5000)
start = time()


def pre_process(row) -> List[str]:
    data = row['steps']
    data = re.sub(r'[^\w\s]', '', data)
    data = data.lower()
    return word_tokenize(data)


df = pd.read_csv('RAW_recipes.csv').sample(100_000)
df = df[['name', 'id', 'steps']]
df['steps'] = df.apply(pre_process, axis=1)

train_df = df.sample(n=int(0.8 * len(df)))
test_df = df.drop(train_df.index)

text = df['steps'].tolist()

for i_gram in [1, 2, 3]:
    train, vocab = padded_everygram_pipeline(i_gram, text)
    temp = []

    lm = MLE(i_gram)
    lm.fit(train, vocab)

    test = test_df['steps'].tolist()
    test = list(list(pad_both_ends(sent, n=i_gram)) for sent in test)
    test = [list(ngrams(sent, n=i_gram)) for sent in test]

    for x in test:
        try:
            temp.append(lm.perplexity(x))
        except ZeroDivisionError:
            winsound.Beep(2500, 500)
            pass
    print('n = ', i_gram, 'perplexity', np.mean(temp))

    print(lm.generate(25, text_seed='<s>'))

print('Elapsed', time()-start)
winsound.Beep(2500, 5000)
