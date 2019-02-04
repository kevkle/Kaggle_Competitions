# Kaggle_Competitions


This is a folder of all the Kaggle_competitions I have participated in 


## Project 1: Elo Merchant Category Recommendation


### Description

The goal is to write an algorithm to predict costumer loyalty based on historical transaction behavior. Elo, one of the most 
important payment brands is working together with merchants in order to give bonuses or discounts to cardholders, who have shopped
at this merchants before.

### Instruction 

1. Download the necessary data into the data folder under (https://www.kaggle.com/c/elo-merchant-category-recommendation/data).
2. Load data into the notebook and go on with the code 


### Packages 

For this project we used the following packages 

```
import numpy as np
import pandas as pd 
import datetime
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold,StratifiedKFold
```




## Project 2: Quora Insincere Questions Classification

### Description

In this competition the task was to classify whether Quora questions were sincere. This includes questions that are racist,homophob or sexist and are willingly asked to provoke. Furthermore, questions that were based on false information are also supposed to be insencere. 

### Instruction 

1. Download the necessary data into the data folder from (https://www.kaggle.com/c/quora-insincere-questions-classification/data).  This includes the files train.csv and test.cvs
2. For the word embeddings I used glove.840B.300d embedding from the website (https://nlp.stanford.edu/projects/glove/)
3. Load data into the notebook and go on with the code 

### Packages 

For this project we used the following packages 

```
import numpy as np
import pandas as pd
import nltk
import re
import time
import random
import matplotlib.pyplot as plt
import sklearn
from numpy import genfromtxt
from tqdm import tqdm
nltk.download('book')
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.book import *
from string import punctuation
from nltk.tokenize import word_tokenize


from keras.models import Model,Sequential
from keras.layers import Dense, Input, Dropout, LSTM, Activation, LeakyReLU,GRU,Flatten,MaxPooling1D,Bidirectional,GlobalMaxPooling1D,Conv1D,Conv2D, MaxPooling2D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.optimizers import Adam
from keras.models import load_model
from sklearn.utils import class_weight
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras import regularizers
```

