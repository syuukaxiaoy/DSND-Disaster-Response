import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
import re
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pickle

nltk.download(['stopwords', 'punkt', 'wordnet', 'averaged_perceptron_tagger'])


def load_data(database_filepath):
    """
    Load data from database and return X, Y, category_names.
    
    Args:
    database_filepath: database filename with path(str)
    
    Return:
      X: messages X(DataFrame)
      y: labels part in messages Y(DataFrame)
      category_names: category-names (str)
    """
    engine = create_engine('sqlite:///' + database_filepath)
    with engine.connect() as conn, conn.begin():
        df = pd.read_sql_table('disaster', conn)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """
    Tokenizes text-data
    
    Args:
    text: Messages as text data(str)
    
    Returns:
    clean_tokens: clean string list
    """
    # normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    #token messages
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    clean_tokens = [WordNetLemmatizer().lemmatize(w) for w in tokens]
    return clean_tokens


def build_model():
    """
    Build model with GridSearchCV
    
    Returns:
    Trained model after performing GridSearch
    """
    # make pipeline
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(DecisionTreeClassifier()))
    ])
    
    # hyper-parameter grid
    parameters = {
        'clf__estimator__max_depth': [5,10,None],
        'clf__estimator__min_samples_leaf': [5,11]
    }
    # make GridSearch
    model = GridSearchCV(pipeline, param_grid=parameters)
    
    return model
    
    


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model
    
    Args:
      model,
      X_test: X test (dataset)
      Y_test: y test (dataset)
      category_names: category names of y
      
    Return
      None
    """
    # predict
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns=category_names)
    
    # print result
    for col in category_names:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
        print("Feature : {}\n".format(col))
        print(classification_report(Y_test[col], y_pred[col]))


def save_model(model, model_filepath):
    """
    Save model as pickle file
    
    Args: 
    model: trained model
    model_filepath: path
   
    """
    # save model as pickle file
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


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
