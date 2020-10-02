import os
import numpy as np
import pandas as pd
import re
import nltk
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import words


class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=False, verbose=True):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        # weights initialization
        print("hello")
        self.theta = np.zeros(X.shape[1])
        self.theta = np.reshape(self.theta,(self.theta.shape[0],1))
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            h = np.reshape(h,(h.shape[0],1))
            # print(self.theta.shape)
            # print(X.shape)
            # print(y.shape)
            # print(h.shape)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient
            
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            loss = self.__loss(h, y)
                
            if(self.verbose ==True and i % 10 == 0):
                print(f'loss: {loss} \t',i+1)
            # print(i)
    
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
    
        return self.__sigmoid(np.dot(X, self.theta))
    
    def predict(self, X):
        return self.predict_prob(X).round()


with open("final_processed_train.pkl", "rb") as input_file:
    x_train = pickle.load(input_file)
with open("final_processed_test.pkl", "rb") as input_file:
    x_test = pickle.load(input_file)
    
print("train shape ",x_train.shape)
print("test shape ",x_test.shape)



tv = TfidfVectorizer(ngram_range = (1,3),sublinear_tf = True,max_features = 40000)
train_tv = tv.fit_transform(x_train)
test_tv = tv.fit_transform(x_test)

X_train = train_tv.toarray()
X_test  = test_tv.toarray()
# print(X.shape)
# print(X[0].shape)
pos_vec = np.ones(3500)
neg_vec = np.zeros(3500)

Y_train = np.concatenate((pos_vec,neg_vec)) 
Y_train = np.reshape(Y_train,(len(Y_train),1))

pos_vec = np.ones(500)
neg_vec = np.zeros(500)

Y_test = np.concatenate((pos_vec,neg_vec)) 
Y_test = np.reshape(Y_train,(len(Y_train),1))

ones = np.ones(X_train.shape[0])
X_train = np.insert(X_train, 0, ones, axis=1)
ones = np.ones(X_test.shape[0])
X_test = np.insert(X_test,0,ones,axis=1)

print("Train shape x:- ",X_train.shape)
print("Train shape y:- ",Y_train.shape)
print("Test shape x:- ",X_test.shape)
print("Test shape y:-",Y_test.shape)

model = LogisticRegression(lr=0.01, num_iter=100)

model.fit(X_train, Y_train)

preds = model.predict(X_test)
count = 0
print(preds[0])
print(Y_test[0])
for i in range(1000):
	if(preds[i] == Y_test[i]):
		count += 1
print(count)
print("Accuracy:- ",(count/1000.0))


