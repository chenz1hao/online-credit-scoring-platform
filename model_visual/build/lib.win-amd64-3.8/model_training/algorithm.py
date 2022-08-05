from mysite import models
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression

def logistic_regression(data_url, train_ratio, random_state):
    df = pd.read_csv(data_url)
    train_X, test_X, train_y, test_y = train_test_split(
        df.drop(df.columns[0], axis=1), df.iloc[:,0], test_size=(100-int(train_ratio)) / 100,
        random_state=int(random_state))

    test_X = np.array(test_X)
    test_y = np.array(test_y)


    lr = LogisticRegression(max_iter=1000000)
    lr.fit(train_X, train_y)
    return lr, test_X, test_y