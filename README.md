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
