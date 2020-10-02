#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import re
import nltk
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
import pickle

# In[2]:


def loadData(base_path):
    pos_files = list()
    neg_files = list()
    
    path = base_path + '/pos/'
    all_files = os.listdir(path)
    for file in all_files:
        fd = open(path+file,'r')
        pos_files.append(fd.readlines())
        print("pos",file)
        if len(pos_files) >= 3500:
        	break;

    path = base_path + '/neg/'
    
    all_files = os.listdir(path)
    for file in all_files:
        fd = open(path+file,'r')
        neg_files.append(fd.readlines())
        print("neg",file)
        if len(neg_files) >= 3500:
        	break
    return pos_files,neg_files


# In[3]:


def preprocess(train_data):
    train_x = list()
    TAG_RE = re.compile(r'<[^>]+>')
    lemmatizer = WordNetLemmatizer() 
    for ind,file in enumerate(train_data):
        text = TAG_RE.sub('', file[0])
        text = re.sub("[^a-zA-Z]", " ", text)
        words = word_tokenize(text) 
        words = [token for token in words if len(token) > 1]
        stopWords = stopwords.words('english')
        stopWords.remove('very')
        stopWords.remove('not')
        words = [word.lower() for word in words if not word.lower() in stopWords]
        final_words = [lemmatizer.lemmatize(w) for w in words]
        train_x.append(" ".join(final_words))
        print(ind)
    return train_x
    


# In[ ]:


train_path = "/home/ajay/Desktop/aclImdb_v1/aclImdb/train"

train_pos,train_neg = loadData(train_path)


# In[28]:


train_x = train_pos + train_neg


# In[33]:


print("# of training Examples :-",len(train_x))


# In[46]:


new_train_x = preprocess(train_x)


# In[49]:


x_train = np.array(new_train_x)


# In[ ]:


print(x_train.shape)


# In[ ]:


with open('final_processed_train.pkl', 'wb') as f:
    pickle.dump(x_train, f)


# In[ ]:




