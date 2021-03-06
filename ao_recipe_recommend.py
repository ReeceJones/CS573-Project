from os import close
import pandas as pd
import numpy as np
import nltk
import ast
# sklearn stuffz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
# nltk stuffz
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
# scipy stuffz
from scipy import spatial
# matplotlib stuffz
import matplotlib.pyplot as plt

def transform(input, ngram_range=(1,1), max_features=200, use_idf=True):
    df = input.copy()
    df = df.dropna().reset_index(drop=True)
    # make everything lowercase, remove quantities and puncuation
    original_names = df.name.copy(deep=True)
    df.name = df.name.str.lower().replace(r'[^a-z ]', '')
    docs = df.name.tolist()
    ingredients = df.ingredients.tolist()
    df.description = df.description.str.lower().replace(r'[^a-z ]', '')
    descriptions = df.description.tolist()
    wnl = WordNetLemmatizer()
    stops = set(stopwords.words('english'))

    for i in range(len(docs)):
        # append ingredients
        docs[i] = docs[i] + ' ' + ' '.join(ast.literal_eval(ingredients[i])) + ' ' + descriptions[i]
        #docs[i] = ' '.join(ast.literal_eval(ingredients[i]))
        # tokenize
        docs[i] = word_tokenize(docs[i])
        # remove morphological affixes
        docs[i] = [wnl.lemmatize(x) for x in docs[i]]
        # rebuild string for tfidf vectorizer
        docs[i] = ' '.join(docs[i])

    vectorizer = TfidfVectorizer(stop_words=stops, ngram_range=ngram_range, max_features=max_features, use_idf=use_idf)
    # compute tfidf values
    name_tfidf = vectorizer.fit_transform(docs)

    print(len(vectorizer.get_feature_names_out()))
    print(name_tfidf.shape)

    # Compute nutrition values
    nutritions = df.nutrition.tolist()
    transformed_nutrition = {
        'n_calories': list(),
        'n_fat': list(),
        'n_sugar': list(),
        'n_sodium': list(),
        'n_protein': list(),
        'n_saturated_fat': list(),
        'n_carbs': list()
    }
    for ent in nutritions:
        unpacked = ast.literal_eval(ent)
        transformed_nutrition['n_calories'].append(unpacked[0])
        transformed_nutrition['n_fat'].append(unpacked[1])
        transformed_nutrition['n_sugar'].append(unpacked[2])
        transformed_nutrition['n_sodium'].append(unpacked[3])
        transformed_nutrition['n_protein'].append(unpacked[4])
        transformed_nutrition['n_saturated_fat'].append(unpacked[5])
        transformed_nutrition['n_carbs'].append(unpacked[6])

    for k,v in transformed_nutrition.items():
        df[k] = v

    # remove transforms columns
    del df['name']
    del df['ingredients']
    del df['description']
    del df['nutrition']

    # remove noise columns
    del df['id']
    del df['contributor_id']
    del df['submitted']
    del df['steps']
    del df['tags']

    feats = vectorizer.get_feature_names_out()
    for i in range(len(feats)):
        df[feats[i]] = name_tfidf[:,i].toarray()

    return df, original_names

def cosine_similarity(corpus, source, titles, n=10):
    for idx, s in source.iterrows():
        closest = list()
        for idx2, row in corpus.iterrows():
            d = spatial.distance.cosine(s, row)
            if len(closest) < n or d < closest[-1][1]:
                closest.append((titles[idx2], d))
                closest.sort(key=lambda x: x[1])
                if len(closest) > n:
                    closest = closest[:n]
        print('----- (cosine similarity)')
        print('source:', titles[idx])
        print('recommendations:')
        for c in closest:
            print(c)
        print('-----')

def kmeans(corpus, source, titles, n=10):
    # make elbow plot
    K = [2,4,8,16,32,64]
    scores = list()
    for k in K:
        model = KMeans(n_clusters=k)
        model.fit(corpus)
        scores.append(model.inertia_)
    plt.plot(K, scores)
    plt.title('KMeans Elbow Plot')
    plt.xlabel('K')
    plt.ylabel('WC-SSD')
    plt.show()
    # K=8 best option
    model = KMeans(n_clusters=8)
    Y = model.fit_predict(corpus)
    print(len(corpus), len(Y))
    # PCA Plot
    r = np.random.random_integers(0, len(corpus)-1, size=250)
    pca = PCA(n_components=2)
    dd = pca.fit_transform(corpus.iloc[r])
    plt.scatter(dd[:,0], dd[:,1], c=Y[r])
    plt.title('PCA Plot w/ K=8')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar()
    plt.show()
    for idx, s in source.iterrows():
        closest = list() # items within same cluster with lowest distance
        for idx2, row in corpus.iterrows():
            d = spatial.distance.euclidean(s, row)
            if Y[idx]==Y[idx2] and (len(closest) < n or d < closest[-1][1]):
                closest.append((titles[idx2], d))
                closest.sort(key=lambda x: x[1])
                if len(closest) > n:
                    closest = closest[:n]
        print('----- (KMeans)')
        print('source:', titles[idx])
        print('recommendations:')
        for c in closest:
            print(c)
        print('-----')


def wcssd(df, assignments):
    a = df.copy()
    a['assignment'] = assignments

    return np.sum(a.groupby('assignment').apply(lambda x: np.sum([np.square(np.array(row[row.index != 'assignment']) - x.loc[:,x.columns != 'assignment'].mean()) for idx, row in x.iterrows()])).tolist())

def agglomerative(corpus, source, titles, n=10):
    K = [2,4,8,16,32,64]
    scores = list()
    for k in K:
        model = AgglomerativeClustering(n_clusters=k)
        print('A')
        Y = model.fit_predict(corpus)
        print('B')
        scores.append(wcssd(corpus, Y))
        print(scores)
    plt.plot(K, scores)
    plt.title('Agglomerative Clustering Elbow Plot')
    plt.xlabel('K')
    plt.ylabel('WC-SSD')
    plt.show()
    # K=8 best option
    model = AgglomerativeClustering(n_clusters=8)
    Y = model.fit_predict(corpus)
    print(len(corpus), len(Y))
    # PCA Plot
    r = np.random.random_integers(0, len(corpus)-1, size=250)
    pca = PCA(n_components=2)
    dd = pca.fit_transform(corpus.iloc[r])
    plt.scatter(dd[:,0], dd[:,1], c=Y[r])
    plt.title('PCA Plot w/ K=8')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar()
    plt.show()
    for idx, s in source.iterrows():
        closest = list() # items within same cluster with lowest distance
        for idx2, row in corpus.iterrows():
            d = spatial.distance.euclidean(s, row)
            if Y[idx]==Y[idx2] and (len(closest) < n or d < closest[-1][1]):
                closest.append((titles[idx2], d))
                closest.sort(key=lambda x: x[1])
                if len(closest) > n:
                    closest = closest[:n]
        print('----- (Agglomerative)')
        print('source:', titles[idx])
        print('recommendations:')
        for c in closest:
            print(c)
        print('-----')

if __name__=='__main__':
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')

    df, titles = transform(pd.read_csv('RAW_recipes.csv'))
    print(df)
    df.to_csv('nutrition_processed_recipes.csv', index=False)

    # Make recommendations
    cosine_similarity(df, df.sample(n=5, random_state=10), titles)
    kmeans(df, df.sample(n=5, random_state=10), titles)
    indexes = np.random.random_integers(0, len(df)-1, 1000)
    agglomerative(df.loc[indexes,:].reset_index(drop=True), df.loc[indexes,:].reset_index(drop=True).sample(n=5, random_state=10), titles[indexes].reset_index(drop=True))