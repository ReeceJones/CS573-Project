import re
import string
import random
import nltk
import pandas as pd
import numpy as np
from typing import List, Set
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter
from nltk.stem import PorterStemmer
from time import time
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB, CategoricalNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve, learning_curve
from sklearn.metrics import classification_report, f1_score, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from collections import Counter

start = time()
vocab = Counter()


def plot_learning_curve(train_sizes, train_scores, test_scores, title, alpha=0.1):
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(train_sizes, train_mean, label='train score', color='blue', marker='o')
    plt.fill_between(train_sizes, train_mean + train_std,
                     train_mean - train_std, color='blue', alpha=alpha)
    plt.plot(train_sizes, test_mean, label='test score', color='red', marker='o')

    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, color='red', alpha=alpha)
    plt.title(title)
    plt.xlabel('Number of training points')
    plt.ylabel('F1-score')
    plt.grid(ls='--')
    plt.legend(loc='best')
    plt.show()


def plot_validation_curve(param_range, train_scores, test_scores, title, alpha=0.1):
    param_range = [x[1] for x in param_range]
    sort_idx = np.argsort(param_range)
    param_range=np.array(param_range)[sort_idx]
    train_mean = np.mean(train_scores, axis=1)[sort_idx]
    train_std = np.std(train_scores, axis=1)[sort_idx]
    test_mean = np.mean(test_scores, axis=1)[sort_idx]
    test_std = np.std(test_scores, axis=1)[sort_idx]
    plt.plot(param_range, train_mean, label='train score', color='blue', marker='o')
    plt.fill_between(param_range, train_mean + train_std,
                 train_mean - train_std, color='blue', alpha=alpha)
    plt.plot(param_range, test_mean, label='test score', color='red', marker='o')
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, color='red', alpha=alpha)
    plt.title(title)
    plt.grid(ls='--')
    plt.xlabel('Weight of class 2')
    plt.ylabel('Average values and standard deviation for F1-Score')
    plt.legend(loc='best')
    plt.show()


df = pd.read_csv('RAW_recipes.csv')
df = df[['name', 'id', 'minutes', 'n_steps', 'steps']]

# print(df['minutes'].describe())
labels = [0, 1, 2]
df['duration'] = pd.cut(df['minutes'], bins=[0, 20, 40, 80], labels=labels, include_lowest=True)
df = df.dropna(subset=['duration'])
stop_words = set(stopwords.words('english'))

stemmer = PorterStemmer()

def pre_process(row, update=True) -> str:
    global vocab

    text = row['steps']
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    text = list(filter(lambda x: x not in stop_words, list(word_tokenize(text))))
    text = list(map(stemmer.stem, text))
    if update:
        for x in text:
            vocab[x] += 1
    return ' '.join(text)



df['steps'] = df.apply(pre_process, axis=1)

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())])
tuned_parameters = {
    'vect__ngram_range': [(1, 1)],
    'tfidf__use_idf': [True],
    'tfidf__norm': ['l2']
}

x_train, x_test, y_train, y_test = train_test_split(df['steps'], df['duration'], test_size=0.2, random_state=42)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
clf = GridSearchCV(text_clf, tuned_parameters, cv=5, scoring='f1_macro', n_jobs=2)
clf.fit(x_train, y_train)
a, b, c = learning_curve(clf, x_train, y_train, cv=5, shuffle=True, n_jobs=2)
plot_learning_curve(a, b, c, title='Learning curve')
# p, q, r = validation_curve(clf, x_test, y_test, param_name='', param_range='', cv=5)
# plot_validation_curve(p, q, r, title='Validation curve')
print(classification_report(y_test, clf.predict(x_test), digits=4))

print('prediction', clf.predict([pre_process({'steps': 'saute capsicum'})]))
print('prediction', clf.predict_proba([pre_process({'steps': 'saute capsicum'})]))

ConfusionMatrixDisplay.from_estimator(clf, x_test, y_test)
plt.show()

print('Elapsed', time()-start, 'seconds')

print(len(vocab))
print(vocab.most_common()[:10])
new_df = pd.DataFrame(columns=['Word', 'frequency', '0', '1', '2'])
for word in vocab.keys():
    s_row = pd.Series([word, vocab[word]] + clf.predict_proba([pre_process({'steps': word}, update=False)])[0].tolist(), index=new_df.columns)
    new_df = new_df.append(s_row, ignore_index=True)
print(new_df.describe())
new_df.to_csv('generated.csv')


"""

Highest probabilities Class 0:
glass
lime
shrimp
sandwich
drink
pineappl
avocado
wedg
blender
color
Highest probabilities Class 1:
muffin
crab
oregano
spoon
skillet
mediumhigh
cou
cornstarch
nonstick
batch
Highest probabilities for Class 2:


"""




