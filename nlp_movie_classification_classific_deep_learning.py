import pandas as pd
import numpy as np
import collections
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding,SimpleRNN,Dense, Flatten


def process_data(file_path): 
	'''
	A function to process data ready for classific machine learning usage.

	:param file_path: input the location of where sauce data was saved. 
	:return: X_train, X_test, X_train_tfidf, X_test_tfidf, y_train, y_test
	'''

	df = pd.read_csv('imdb_master.csv', encoding = "ISO-8859-1")

	df.loc[df['label'] == 'neg', 'label'] = 0
	df.loc[df['label'] == 'pos', 'label'] = 1
	df.loc[df['label'] == 'unsup', 'label'] = np.nan


	# we can only select texts already have labels

	df_train = df[(df['type'] == 'train')& (~df['label'].isnull())]

	X_train = df_train['review']
	y_train = df_train['label']

	X_test = df[df['type'] == 'test']['review']
	y_test = df[df['type'] == 'test']['label']

	y_train = np.array(y_train).astype('int')
	y_test = np.array(y_test).astype('int')

	y_train = y_train.reshape(-1,1).ravel()
	y_test = y_test.reshape(-1,1).ravel()


	vectorizer = TfidfVectorizer(stop_words = 'english', encoding = "ISO-8859-1")
	vectorizer.fit(X_train)

	X_train_tfidf = vectorizer.transform(X_train)
	X_test_tfidf = vectorizer.transform(X_test)

	return  X_train, X_test, X_train_tfidf, X_test_tfidf, y_train, y_test


def training(classifier, X_train, X_test, y_train, y_test):
	'''
	A function to fit classific machine learning classifiers to the processed data

	:param classifier: classific machine learning classifiers, including logistic regression, random forest, and gradient boosting, etc
	:param X_train: input data for training
	:param X_test: input data for testing
	:param y_train: label data for training
	:param y_test: label data for testing 
	:return: X_train, X_test, X_train_tfidf, X_test_tfidf, y_train, y_test
	'''

    model = classifier

    model.fit(X_train, y_train)

    y_test_predict = model.predict(X_test)

    report = classification_report(y_test, y_test_predict)

    print(report)

    return model


def data_process_for_deep_learning(X_train, X_test):
	'''
	A function to tokenize and prepare data to sequence, reading for deep learning models, such as Simple RNN and LSTM. 

	:param y_test: label data for testing 
	:return: X_train_sequence, X_test_sequence
	'''

	# Let's use the most common 6000 words and disregard other non-common words. Also called as vocabulary size

	# Keras tokenizer basically just assign a value for each word. no count. no meaning. 

	max_features = 6000

	keras_tokenizer = Tokenizer(num_words=max_features)

	keras_tokenizer.fit_on_texts(X_train)

	X_train_keras_tokenized = tokenizer.texts_to_sequences(X_train)

	X_test_keras_tokenized = tokenizer.texts_to_sequences(X_test)


	# Assume the maximum length for a review is 500 words

	max_words_in_review = 500

	X_train_sequence = pad_sequences(X_train_keras_tokenized, maxlen = max_words_in_review)

	X_test_sequence = pad_sequences(X_test_keras_tokenized, maxlen = max_words_in_review)

	return X_train_sequence, X_test_sequence


def getting_deep_model(X_train_sequence, X_test_sequence, y_train, y_test, classifier):
	'''
	A function to train and fit data to deep learning models, such as RNN and LSTM.

	:param X_train_sequence: processed sequential X_train data for training 
	:param X_test_sequence: processed sequential X_test data for testing
	:param y_train: label data for training 
	:param y_test: label data for testing 
	:param classifier: input classifiers for training - such as RNN, LST, 
	:return: after trained model
	'''

	max_features = 6000
	max_words_in_review = 500
	# embed_size = 100

	# glove embedding and some others all use 100 as the embedding dimension

	model = Sequential()
	model.add(Embedding(input_dim = max_features, output_dim = embed_size))
	model.add(classifier) 
	model.add(Dense(1,activation='sigmoid'))
	model.summary()
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	batch_size = 128
	# normally you want a batch size that's a power of 2

	epochs = 3

	model.fit(X_train_sequence, y_train, epochs=epochs, batch_size = batch_size, validation_split=0.2)

	y_test_predict = model.predict(X_test_sequence)
	y_test_predict = y_test_predict>0.5

	print('accuracy score is', accuracy_score(y_test_predict, y_test))
	print('f1 score is', f1_score(y_test_predict, y_test))

	return model



# getting data
X_train, X_test, X_train_tfidf, X_test_tfidf, y_train, y_test = process_data('imdb_master.csv')

# classific machine learning models
LogisticRegressionModel = training(LogisticRegression(), X_train_tfidf, X_test_tfidf, y_train, y_test)

RandomForestModel = training(RandomForestClassifier(), X_train_tfidf, X_test_tfidf, y_train, y_test)

GradientBoostingModel = training(GradientBoostingClassifier(), X_train_tfidf, X_test_tfidf, y_train, y_test)


# process data for deep learning
X_train_sequence, X_test_sequence = data_process_for_deep_learning(X_train, X_test)

# deep machine learning models
SimpleRnnModel = getting_deep_model(X_train_sequence, X_test_sequence, y_train, y_test, SimpleRNN(100))

LSTMModel = getting_deep_model(X_train_sequence, X_test_sequence, y_train, y_test, LSTM(100,dropout=0.3))




