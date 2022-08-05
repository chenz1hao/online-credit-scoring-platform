import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression



def risk_slim(data_url, train_ratio, random_state):
    pred_y, pred_y_prob, test_y, solutions = riskslim_in_use.run(data_url, train_ratio, random_state)

    return pred_y, pred_y_prob, test_y, solutions

def logistic_regression(data_url, train_ratio, random_state, label_name, feature_name_list):
    df = pd.read_csv(data_url)
    train_X, test_X, train_y, test_y = train_test_split(
        df[feature_name_list], df[label_name], test_size=(100-int(train_ratio)) / 100,
        random_state=int(random_state))
    test_X = np.array(test_X)
    test_y = np.array(test_y)



    lr = LogisticRegression(max_iter=1000000)
    lr.fit(train_X, train_y)
    return lr, test_X, test_y




