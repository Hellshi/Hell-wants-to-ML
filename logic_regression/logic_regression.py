import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression

with open ('risco_credito.pkl', 'rb') as f:
    X_risco_credito, y_risco_credito = pickle.load(f)

print(len(X_risco_credito), len(y_risco_credito))

#keeping only  alto and baixo
X_risco_credito = np.delete(X_risco_credito, [2,7,11], axis=0)
y_risco_credito = np.delete(y_risco_credito, [2,7,11], axis=0)

print(len(X_risco_credito), len(y_risco_credito))

#Training database

logistic_credit_risk = LogisticRegression(random_state=1)
logistic_credit_risk.fit(X_risco_credito, y_risco_credito)

print(logistic_credit_risk.intercept_, logistic_credit_risk.coef_)

predict = logistic_credit_risk.predict([[0,0,1,2], [2,0,0,0]])
print(predict)