import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
import re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from joblib import dump

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def load_data(database_filepath):
    """
    loads disaster pipeline data from a database_filepath and 
    divides in X, Y and categories
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table(database_filepath, engine)  
    
    # prepares to numpy
    X = df[['message']].values.ravel()

    categories = ['related', 'request', 'offer',
           'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
           'security', 'military', 'child_alone', 'water', 'food', 'shelter',
           'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
           'infrastructure_related', 'transport', 'buildings', 'electricity',
           'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
           'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
           'other_weather', 'direct_report']
    
    # prepares to numpy
    Y = df[categories].values
    return X, Y, categories


def tokenize(text):
    """
        receives a String text from disaster pipeline and returns lemmatized, normatized tokens
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower().strip())
    
    tokens = word_tokenize(text)
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]
    return tokens


def build_model():
    """
        creates an model to classify in 36 categories the text from disaster pipeline problem
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__bootstrap': [True],
        'clf__estimator__max_depth': [80, 90, 100, 110],
        'clf__estimator__max_features': [2, 3],
        'clf__estimator__min_samples_leaf': [3, 4, 5],
        'clf__estimator__min_samples_split': [8, 10, 12],
        'clf__estimator__n_estimators': [100, 200, 300, 1000]
    }
    
    return GridSearchCV(pipeline, parameters)
    

def evaluate_model(model, X_test, Y_test, category_names):
    """
     evaluates the model in the test data
    """
    print(classification_report(Y_test, model.predict(X_test)))


def save_model(model, model_filepath):
    """
        save model to a pickle file
    """
    dump(model, model_filepath)

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