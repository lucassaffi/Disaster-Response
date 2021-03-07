import sys
import pandas as pd
from sqlalchemy import create_engine
import string
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import pickle
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('wordnet')

def load_data(database_filepath):
    # load data from database
    df = pd.read_sql_table('MySQL',database_filepath)
    X = df['message']
    y = df[df.columns[4:]]

    return X,y,y.columns


def tokenize(text):
    # transform to lower case
    text = text.lower()

    # Remove punctuation and split
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(text)

    # Reduce inflectional forms to a common base form
    lemmatizer=nltk.WordNetLemmatizer()

    list_words = []
    for word in words:
        list_words.append(lemmatizer.lemmatize(word))

    return list_words


def build_model(X):
    # Defining vocabulary used for all answers
    vocabulary = []
    word_messages = X.apply(lambda x: tokenize(x))

    for messages in word_messages:
        for word in messages:
            vocabulary.append(word)

    vocabulary = pd.Series(vocabulary).unique()

    # Setting Pipeline
    pipeline = Pipeline([('vect',CountVectorizer(tokenizer=tokenize,vocabulary=vocabulary)),
                         ('tfidf',TfidfTransformer()),
                         ('rfc',MultiOutputClassifier(RandomForestClassifier()))])

    # Setting parameter for Gridsearch. It is taking a long time to test all possibilities, even jut for one column ("related")
    parameters = {'rfc__estimator__n_estimators':[1],'rfc__estimator__min_samples_split':[2]}

    cv = GridSearchCV(pipeline,param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)

    # Split all predicted/test columns
    y_test_columns = np.hsplit(Y_test,36)
    y_pred_columns = np.hsplit(Y_pred,36)

    for i in range(len(y_test_columns)):
        print(classification_report(y_test_columns[i],y_pred_columns[i]))


def save_model(model, model_filepath):
    pickle.dump(model,open(model_filepath,"wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model(X)

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
