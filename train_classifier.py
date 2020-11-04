import sys
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)
import sys
import os
import re
from sqlalchemy import create_engine
import pickle
from scipy.stats import gmean
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.metrics import precision_recall_fscore_support, classification_report

def load_data(database_filepath):
    """
    Load Data Function
    Arguments:
    database_filepath -> path to SQLite db
    Output:
    X -> feature DataFrame
    Y -> label DataFrame
    category_names -> used for data visualization (app)
    """
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterResponse',engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X, Y, category_names

def tokenize(text):
    """
    Tokenize the text strings
    Args: text string
    Return:
    clean_tokens: Array of tokenized strings
    """
    # Remove punctuation characters   
    punctuation_characters = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_characters = re.findall(punctuation_characters, text)
    for ch in detected_characters :
        text = text.replace(ch, "") 
        #convert to lower
        text = text.lower()
        # Extract the word tokens from the provided text
        tokens = word_tokenize(text)
        #Lemmanitizer to remove inflectional and derivationally related forms of a word
        lemmatizer = WordNetLemmatizer()
        # List of clean tokens
        clean_tokens =[]
        for tok in tokens:
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)
            return clean_tokens 
        
      
# Build a custom transformer which will extract the starting verb of a sentence
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class
    
    This class extract the starting verb of a sentence,
    creating a new feature for the ML classifier
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    # Given it is a tranformer we can return the self 
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

    

def build_model():
    """
    Build Model function
    
    This function output is a Scikit ML Pipeline that process text messages
    according to NLP best-practice and apply a classifier.
    """
    # create pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=AdaBoostClassifier()))
    ])
    parameters = {
        'vect__min_df':[1,10,50],
        'clf__estimator__learning_rate': [0.001, 0.01, 0.1],
        'tfidf__smooth_idf': [True, False]
    }
    model  = GridSearchCV(pipeline, param_grid=parameters, cv=2) 
    return model 

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate Model function
    This function applies ML pipeline to a test set and prints out
    model performance (accuracy and f1score)
    Arguments:
    model -> Scikit ML Pipeline
    X_test -> test features
    Y_test -> test labels
    category_names -> label names (multi-output)
    """
    # predict on test data
    print("Model Evaluation")
    y_pred = model.predict(X_test)
    i = 0
    for column in category_names:
        print('\n')
        print('-----','Evaluation of ' , column, ': -----')
        print(classification_report(Y_test[column], y_pred[:,i]), '\n')
        print('\n')
        i += 1
    pass


def save_model(model, model_filepath):
    """
    Save Model function
    This function saves trained model as Pickle file, to be loaded later
    Arguments:
    model -> GridSearchCV or Scikit Pipelin object
    model_filepath -> destination path to save .pkl file
    """
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))
    pass

def main():
    """
    Train Classifier Main function
    This function applies the Machine Learning Pipeline:
    1) Extract data from SQLite db
    2) Train ML model on training set
    3) Estimate model performance on test set
    4) Save trained model as Pickle
    """
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
       # evaluate_model(model, X_test, Y_test, category_names)

        #print('Saving model...\n    MODEL: {}'.format(model_filepath))
        #save_model(model, model_filepath)

        #print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()