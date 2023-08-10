"""Implement ML methods with cross-validation."""

from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def cv_logistic_reg(X, y, cv=10):
    lr = LogisticRegression(C=0.01, solver='liblinear')
    lr_cv = cross_validate(lr, X, y, cv=cv, scoring=('accuracy', 'roc_auc', 'recall','precision'), 
                           return_train_score=True)    
    return get_metrics_mean(lr_cv)
   
def cv_kneighbors(X, y, cv=10):
    k = 4
    kn = KNeighborsClassifier(n_neighbors = k)
    kn_cv = cross_validate(kn, X, y, cv=cv, scoring=('accuracy', 'roc_auc', 'recall','precision'), 
                           return_train_score=True)
    return get_metrics_mean(kn_cv)
    
def cv_svc(X, y, cv=10):
    sv = SVC(kernel='rbf', probability=True)
    sv_cv = cross_validate(sv, X, y, cv=cv, scoring=('accuracy', 'roc_auc', 'recall','precision'), 
                           return_train_score=True)
    return get_metrics_mean(sv_cv)
    
def cv_rf(X, y, cv=10):
    rf = RandomForestClassifier()
    rf_cv = cross_validate(rf, X, y, cv=cv, scoring=('accuracy', 'roc_auc', 'recall','precision'), 
                           return_train_score=True)
    return get_metrics_mean(rf_cv)

def get_metrics_mean(cv):
    
    # Return the mean of the cross-validation metrics for the test data. 
    accuracy_test = cv['test_accuracy'].mean()
    roc_auc_test = cv['test_roc_auc'].mean()
    recall_test = cv['test_recall'].mean()
    precision_test = cv['test_precision'].mean()    
    return [accuracy_test, roc_auc_test, recall_test, precision_test]

