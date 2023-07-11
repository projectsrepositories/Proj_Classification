from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import analysis

# Implementation of the machine learning methods
def ml_logistic_reg(X_train, X_test, y_train, y_test, yhats):
    lr = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
    yhat_lr = lr.predict(X_test)
    yhat_prob_lr = lr.predict_proba(X_test)
    yhats.append(yhat_lr)
    metrics = analysis.evaluate_models(y_test, yhat_lr, yhat_prob_lr)
    return [metrics, yhats]
   
def ml_kneighbors(X_train, X_test, y_train, y_test, yhats):
    k = 4
    kn = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
    yhat_kn = kn.predict(X_test)
    yhat_prob_kn = kn.predict_proba(X_test)
    yhats.append(yhat_kn)
    metrics = analysis.evaluate_models(y_test, yhat_kn, yhat_prob_kn)
    return [metrics, yhats]
    
def ml_svc(X_train, X_test, y_train, y_test, yhats):
    sv = SVC(kernel='rbf', probability=True).fit(X_train, y_train)
    yhat_sv = sv.predict(X_test)
    yhat_prob_sv = sv.predict_proba(X_test)
    yhats.append(yhat_sv)
    metrics = analysis.evaluate_models(y_test, yhat_sv, yhat_prob_sv)
    return [metrics, yhats]
    
def ml_rf(X_train, X_test, y_train, y_test, yhats):
    rf = RandomForestClassifier().fit(X_train, y_train)
    yhat_rf = rf.predict(X_test)
    yhat_prob_rf = rf.predict_proba(X_test)
    yhats.append(yhat_rf)
    metrics = analysis.evaluate_models(y_test, yhat_rf, yhat_prob_rf)
    return [metrics, yhats]

