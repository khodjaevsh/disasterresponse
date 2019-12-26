import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer 
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import AdaBoostClassifier
from joblib import dump, load

# Count the number of tokens 
class TextLength(BaseEstimator, TransformerMixin):

    def text_len_count(self, text):
        tokens = word_tokenize(re.sub('[^0-9a-zA-Z]',' ',text).strip())
        tokens = [x for x in tokens if x not in stopwords.words('english')]
        text_length = len(tokens)

        return text_length

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_text_len = pd.Series(X).apply(self.text_len_count)
        return pd.DataFrame(X_text_len)
        
def load_data(database_filepath):
    """
    INPUT:
        database_filepath (string) : database location
    OUTPUT:
        X (np.array) : messages to process
        y (np.array) : training/evaluating categories
        labels (np.array) : list of message classification labels
    """
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_data', engine)
    X = df.message
    Y = df.iloc[:,4:]
    category_names = (df.drop(['id', 'message','original','genre'],
                      axis=1)).columns.values
    
    return X, Y, category_names


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

    
def build_model():
   
    pipeline_adaboost = Pipeline([
    ('features', FeatureUnion([
        ('text_prep', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize, ngram_range = (1,2))),
            ('tfidf', TfidfTransformer(use_idf = False)),
        ])),
        ('text_len', TextLength())
    ])),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    return pipeline_adaboost


def evaluate_model(model, X_test, Y_test, category_names):
    preds = model.predict(X_test)
    
    for i,category in enumerate(category_names):
        print('Message category:', category)
        print(classification_report(Y_test.iloc[:,i], [row[i] for row in preds] ))
        print('-------------' * 8)

def save_model(model, model_filepath):
    """
    Save model to a pickle file
    """
    dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
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