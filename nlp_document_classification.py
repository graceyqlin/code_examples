# Document_Category
# https://www.hackerrank.com/challenges/document-classification/problem
# Your task is to classify documents into one of eight categories: [1,2,3,...8]


# Sample Input

# 3
# This is a document
# this is another document
# documents are seperated by newlines

# Sample Output

# 1
# 4
# 8



import sys
from sklearn.feature_extraction import text
from sklearn import pipeline
from sklearn import linear_model
import numpy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit

def load_training_data(filename):
    '''
    A function to load and process data, getting it ready for classific machine learning usage.

    :param file_path: input the location of where sauce data was saved. 
    :return: x_train, y_train, x_validation, y_validation
    '''

    df = pd.read_fwf(filename, delim_whitespace = True, header = None)
    df = df.rename(columns = {df.columns[0]:'text'})
    df = df[['text']][1:]
    df = pd.DataFrame(df['text'].str.split(" ", 1).tolist(), columns = ['category','text'])
    train_set, validation_set = train_test_split(df, test_size=.2)
    y_train = train_set['category']
    x_train = train_set['text']
    y_validation = validation_set['category']
    x_validation = validation_set['text']

    return x_train, y_train, x_validation, y_validation

def load_new_input_data(filename):
    '''
    A function to load data directly from Hackerrank server. 
    Process data to just get text information

    :param file_path: input the location of where sauce data was saved. 
    :return: x, which is a list of text
    '''
    df = pd.read_fwf(filename, delim_whitespace = True, header = None)
    df = df.rename(columns = {df.columns[0]:'text'})
    df = df[['text']][1:]
    x = df['text']
    
    return x

def vectorize_training_data(x_train):
    '''
    A function to vectorize the training data. 

    :param x_train: input the data in text format 
    :return: x_train_vectorized, which is vectorized training data, and vectorizer, which is the vectorizer after data fitting
    '''
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 1),
                          strip_accents='ascii', lowercase=True)
    x_train_vectorized = vectorizer.fit_transform(x_train)
    
    return x_train_vectorized, vectorizer   


def vectorize_testing_data(x_test, vectorizer):
    '''
    A function to vectorize the testing data. 

    :param x_test: input the data in text format 
    :param vectorizer: the vectorizer after data fitting
    :return: x_test_vectorized, which is vectorized text data
    '''
    x_test_vectorized = vectorizer.transform(x_test)
    
    return x_test_vectorized

def train_model(model,x_train_vectorized, y_train):
    '''
    A function to train a model

    :param model: a classifier before training
    :param x_train_vectorized: vectorized data for training
    :param y_train: labeled data for training
    :return: classifier after training and data fitting
    '''
    classifier = linear_model.SGDClassifier(class_weight='balanced')
    classifier.fit(x_vectorized, y_train)
    
    return classifier

def evaluate(model, x_validation_vectorized, y_validation):
    '''
    A function to evaluate the model performance

    :param model: classifier after training and data fitting
    :param x_validation_vectorized: vectorized data for validation
    :param y_validation: labeled data for validation
    :return: classification_report, accuracy_score to read and compare model performance
    '''

    predictions = model.predict(x_validation_vectorized)
    classification_report =  (classification_report(y_validation, predictions))
    accuracy_score =  ("The accuracy score is {:.2%}".format(accuracy_score(y_validation, predictions)))

    return classification_report, accuracy_score

def find_best_parameter(input_model, param_grid, x_train_vectorized, y_train):
    '''
    A function to find the best parameters using grid search

    :param input_model: classifier after training and data fitting
    :param param_grid: vectorized data for validation
    :param x_train_vectorized: labeled data for validation
    :param y_train: labeled data for validation
    :return: classification_report, accuracy_score to read and compare model performance
    '''
    cv_sets = ShuffleSplit(n_splits = 2, test_size = .33, random_state = 1)
    grid_search = GridSearchCV(estimator=input_model, param_grid=param_grid, cv = cv_sets, scoring='accuracy')
    grid_search.fit(x_train_vectorized, y_train)
    best_model = grid_search.best_estimator_
    
    return best_model

x_train, y_train, x_validation, y_validation = load_training_data('trainingdata.txt')

x_train_vectorized, vectorizer = vectorize_training_data(x_train)

x_validation_vectorized = vectorize_testing_data(x_validation, vectorizer)

model1 = LogisticRegression()
model = train_model(model1,x_train_vectorized, y_train)
print(evaluate(model, x_validation_vectorized, y_validation))


# grid search for best C
param_grid={'C': [0.01, 0.1, 1, 10]}
model = find_best_parameter(model1, param_grid, x_train_vectorized, y_train)
print(evaluate(model, x_validation_vectorized, y_validation))

model2 = linear_model.SGDClassifier()
model = train_model(model2,x_train_vectorized, y_train)
print(evaluate(model, x_validation_vectorized, y_validation))

model3 = MultinomialNB()
model = train_model(model3,x_train_vectorized, y_train)
print(evaluate(model, x_validation_vectorized, y_validation))

x_test = load_new_input_data('stdin.txt')

x_test_vectorized = vectorize_testing_data(x_test, vectorizer) 


for line in model.predict(x_test_vectorized):
    print(line)



# # when we are running in hacker rank:
# import fileinput 

# def load_hacker_rank_input():
#     temp = []  
    
#     for f in fileinput.input(): 
#         temp.append(f)
    
#     df = pd.DataFrame(temp)
#     df = df.rename(columns = {df.columns[0]:'text'})
#     df['text'] = df['text'].str.replace('\n', '')
#     df = df[['text']][1:]
#     x = df['text']
#     return x

# x_test = load_hacker_rank_input()

# x_test_vectorized = vectorize_testing_data(x_test, vectorizer)

# for line in model.predict(x_test_vectorized):
#     print(line)



