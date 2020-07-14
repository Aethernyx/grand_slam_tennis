import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

match_data = 'men_match_features.csv'
men_matches = pd.read_csv(match_data)

X = men_matches.drop(['match_winner', 'match_id', 'Unnamed: 0'], axis=1)
y = men_matches['match_winner']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

log_model = LogisticRegression(max_iter=250)

log_model.fit(X_train, y_train)

log_pred = log_model.predict(X_test)

log_proba = log_model.predict_proba(X_test)

pickle.dump(log_model, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))