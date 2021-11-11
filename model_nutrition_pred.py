import pandas as pd
import nltk
import ast
# sklearn stuffz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
# nltk stuffz
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


if __name__=='__main__':
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')

    df = pd.read_csv('RAW_recipes.csv')
    df.dropna(inplace=True)
    # make everything lowercase, remove quantities and puncuation
    df.name = df.name.str.lower().replace(r'[^a-z ]', '')
    docs = df.name.tolist()
    ingredients = df.ingredients.tolist()
    wnl = WordNetLemmatizer()
    stops = set(stopwords.words('english'))

    for i in range(len(docs)):
        # append ingredients
        docs[i] = docs[i] + ' ' + ' '.join(ast.literal_eval(ingredients[i]))
        #docs[i] = ' '.join(ast.literal_eval(ingredients[i]))
        # tokenize
        docs[i] = word_tokenize(docs[i])
        # remove morphological affixes
        docs[i] = [wnl.lemmatize(x) for x in docs[i]]
        # rebuild string for tfidf vectorizer
        docs[i] = ' '.join(docs[i])

    vectorizer = TfidfVectorizer(stop_words=stops, ngram_range=(1,1), max_features=200, use_idf=True)
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
    del df['nutrition']

    # remove noise columns
    del df['id']
    del df['description']
    del df['contributor_id']
    del df['submitted']
    del df['steps']
    del df['tags']

    feats = vectorizer.get_feature_names_out()
    for i in range(len(feats)):
        df[feats[i]] = name_tfidf[:,i].toarray()

    print(df)
    df.to_csv('nutrition_processed_recipes.csv', index=False)

    # make KNN possible
    df = df.sample(frac=0.2)

    # Split dataset
    train = df.sample(frac=0.8)
    test = df[~df.index.isin(train.index)]

    # Remove predicted columns
    train_y = train.n_calories
    test_y = test.n_calories

    del train['n_calories']
    del train['n_fat']
    del train['n_sugar']
    del train['n_sodium']
    del train['n_protein']
    del train['n_saturated_fat']
    del train['n_carbs']

    del test['n_calories']
    del test['n_fat']
    del test['n_sugar']
    del test['n_sodium']
    del test['n_protein']
    del test['n_saturated_fat']
    del test['n_carbs']

    # Learn
    X = train.to_numpy()
    y = train_y.to_numpy()
    #reg = LassoCV(cv=5).fit(X, y)
    #reg = LinearRegression().fit(X,y)
    reg = KNeighborsRegressor(n_neighbors=3, algorithm='brute').fit(X,y)
    #print(reg.coef_)
    print(reg.score(X, y))

    X_t = test.to_numpy()
    y_t = test_y.to_numpy()
    print(reg.score(X_t, y_t))
    print('Preds:\n',reg.predict(X_t))
    print('Real:\n', y_t)