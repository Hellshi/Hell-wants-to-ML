import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix


##Credit base
with open ('risco_credito.pkl', 'rb') as f:
    X_risco_credito, y_risco_credito = pickle.load(f)

""" print(len(X_risco_credito), len(y_risco_credito)) """

#keeping only  alto and baixo
X_risco_credito = np.delete(X_risco_credito, [2,7,11], axis=0)
y_risco_credito = np.delete(y_risco_credito, [2,7,11], axis=0)

""" print(len(X_risco_credito), len(y_risco_credito)) """

#Training database

logistic_credit_risk = LogisticRegression(random_state=1)
logistic_credit_risk.fit(X_risco_credito, y_risco_credito)

""" print(logistic_credit_risk.intercept_, logistic_credit_risk.coef_) """

predict = logistic_credit_risk.predict([[0,0,1,2], [2,0,0,0]])
""" print(predict) """

#Yet ANOTHER credit base (Yes I hate, I mean, I love my code XD)

with open ('credit.pkl', 'rb') as f:
    X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(f)

print(X_credit_treinamento.shape, y_credit_treinamento.shape)

logistic_credit = LogisticRegression(random_state=1)
logistic_credit.fit(X_credit_treinamento, y_credit_treinamento)

print(logistic_credit.intercept_, logistic_credit.coef_)

credit_predict = logistic_credit.predict(X_credit_teste)
print(accuracy_score(y_credit_teste, credit_predict))

#Classification Report
print(classification_report(y_credit_teste, credit_predict))

#Confusion Matrix
cm = ConfusionMatrix(logistic_credit)
cm.fit(X_credit_treinamento, y_credit_treinamento)
cm.score(X_credit_treinamento, y_credit_treinamento)
cm.show()


