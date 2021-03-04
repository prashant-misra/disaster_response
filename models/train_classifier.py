import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])

# import libraries
import pandas as pd
import pickle
import re
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, classification_report, accuracy_score, f1_score


def load_data(database_filepath):
    
    '''
    Load data from database as dataframe
    Input:
        database_filepath: File path of sql database
    Output:
        X: Message data (features)
        Y: Categories (target)
        category_names: Labels for 36 categories
    '''
        
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Disaster_Pipeline_Data',con=engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    
    return X,Y,category_names


def tokenize(text):
    
    '''
    Tokenize and Lemmatize text
    Input:
        text: original text message
    Output:
        tokens: Tokenized, Normalized and lemmatized text
    '''
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", ' ', text.lower())
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Listing Stopwords from the text
    stop_words = stopwords.words("english")
    
    # Initializing Lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Lemmatizing words except stopwords
    clean_tokens = [lemmatizer.lemmatize(tok).lower() for tok in tokens if tok not in stop_words]
    
    return tokens


def build_model():
    '''
    Build the ML Model
    Input:
        None
    Output:
        model: Return thw ML model
    '''
    # Declaring Pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # Declaring parameters
    parameters = {#'clf__estimator__n_estimators': [50, 100,200],
        'tfidf__use_idf': (True, False),
                  'clf__estimator__min_samples_split': [2, 3, 4]
                 }

    
    # Declaring GRidSearch
    cv = GridSearchCV(pipeline,parameters)
    
    return cv
    
    


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    
    for i in range(len(category_names)):
        print("Category:", category_names[i],"\n", classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, "wb"))


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