"""Perform binary classification.

Parameters for binary classification can be set in the file /src/params.py.

Outputs of binary classification:
1. Performance evaluation output location: /output/
2. Plots output location: /plots/
"""

import numpy as np
import pandas as pd
import csv
import methods_ml, methods_cv, params
import preprocess, visualization
import warnings
warnings.filterwarnings("ignore")

model_names = ['LR', 'RF', 'KN', 'SVM']
titles = ['Accuracy', 'ROC_AUC', 'Recall', 'Precision']
np.random.seed(params.seed)

def random_split():  
    """Check the out-of-sample predictive performance using random-split.
    
    Perform random-split to select random test set in each iteration.
    
    Returns
    -------
    list: A list of dataframes containing averages of the performance metrics.
    """
    
    df_acc = pd.DataFrame(columns=model_names)
    df_auc = pd.DataFrame(columns=model_names)
    df_rec = pd.DataFrame(columns=model_names)
    df_pre = pd.DataFrame(columns=model_names)

    df_list = [df_acc, df_auc, df_rec, df_pre]
    df_avg_list = []  # List of dataframes
    yhats = []
    filenames = ['accuracy', 'auc', 'recall', 'precision']

    # Perform binary classification for num_run iterations.
    for i in range(params.num_run):
        
        # Data i.e. X_train, X_test, y_train, y_test are selected in each iteration
        data = preprocess.split_data(X, y, params.test_size)

        # yhats is reset in each iteration but acc, auc, rec, pre, and pre are appended
        *df_list, yhats = iteration(i, data, *df_list)

    # Save all the evaluation metric values to the files and calculate their averages.
    for i in range(len(df_list)):
        df_list[i].to_csv(f"../output/{filenames[i]}{params.num_run}.csv", index=False)        
        df_avg_list.append(df_list[i].mean(axis=0).to_frame())
        df_avg_list[i].columns = [f'Avg{params.num_run}']
    
    # Create boxplots for accuracy, roc-auc, recall and precision.
    print('\n\nPredictive performance of the ML methods for the randomly selected '
           'test sets in each iteration')
    visualization.plot_boxplot(df_list, titles, params.num_run)
    
    # Plot confusion matrix for the last iteration only using yhats.
    print('\n\nConfusion matrix for the last iteration of the randomly selected test sets. ')
    visualization.plot_confusion_matrix(data[3], yhats, model_names, params.labels, params.num_run)      
    return df_avg_list

def iteration(iterno, data, df_acc, df_auc, df_pre, df_rec):    
    yhats = []
    metrics_lr, yhats = methods_ml.ml_logistic_reg(*data, yhats)
    metrics_rf, yhats = methods_ml.ml_rf(*data, yhats)
    metrics_kn, yhats = methods_ml.ml_kneighbors(*data, yhats)
    metrics_svc, yhats = methods_ml.ml_svc(*data, yhats)
    
    df_acc_new = pd.DataFrame([{'LR': metrics_lr['acc'], 'RF': metrics_rf['acc'], 
                               'KN': metrics_kn['acc'], 'SVM': metrics_svc['acc']}])
    df_auc_new = pd.DataFrame([{'LR': metrics_lr['auc'], 'RF': metrics_rf['auc'], 
                               'KN': metrics_kn['auc'], 'SVM': metrics_svc['auc']}])
    df_rec_new = pd.DataFrame([{'LR': metrics_lr['rec'], 'RF': metrics_rf['rec'], 
                               'KN': metrics_kn['rec'], 'SVM': metrics_svc['rec']}])
    df_pre_new = pd.DataFrame([{'LR': metrics_lr['pre'], 'RF': metrics_rf['pre'], 
                               'KN': metrics_kn['pre'], 'SVM': metrics_svc['pre']}])
    
    # Add new row for this iteration to the dataframes
    df_acc = pd.concat([df_acc, df_acc_new]) 
    df_auc = pd.concat([df_auc, df_auc_new]) 
    df_rec = pd.concat([df_rec, df_rec_new]) 
    df_pre = pd.concat([df_pre, df_pre_new]) 
    return [df_acc, df_auc, df_rec, df_pre, yhats]

def cross_validation(list_df_metrics):
    """Check the out-of-sample predictive performance using cross-validation.
    
    Parameters
    ----------
    list_df_metrics : list
        A list of dataframes for averages of the performance metrics of the 
        random-split method.      
    """
    
    # Perform cross-validation to check the out-of-sample predictive performance.
    metrics_lr = methods_cv.cv_logistic_reg(X, y, params.cv)
    metrics_rf = methods_cv.cv_rf(X, y, params.cv)
    metrics_kn = methods_cv.cv_kneighbors(X, y, params.cv)
    metrics_sv = methods_cv.cv_svc(X, y, params.cv)
    
    filenames = [f'cv{params.cv}_numrun{params.num_run}_acc_Avg.csv', 
                 f'cv{params.cv}_numrun{params.num_run}_auc_Avg.csv', 
                 f'cv{params.cv}_numrun{params.num_run}_rec_Avg.csv', 
                 f'cv{params.cv}_numrun{params.num_run}_pre_Avg.csv']
    
    # Convert ML-methods-wise lists to metrics-wise lists and append as a new column in df
    zipped_list = list(zip(metrics_lr, metrics_rf, metrics_kn, metrics_sv))    
    for i in range(len(zipped_list)):
        list_df_metrics[i][f'CV{params.cv}'] = zipped_list[i]    
        list_df_metrics[i].to_csv(f"../output/ {filenames[i]}")
    print('\n\nComparison of the average predictive performances of the two methods:')
    print('1: Randomly selecting test set in each of the n iterations.')
    print('   AVG10 means average of 10 iterations.')
    print('2: Performing k-fold cross-validation.')
    print('   CV5 means 5-fold cross-validation.')
    
    # Plot bar charts to compare the performances of the CV and random-split methods.
    visualization.plot_bar_charts(list_df_metrics, titles)

# Read and preprocess data for modeling and prediction.
X, y = preprocess.get_data()
X = preprocess.normalize_data(X, X)

# Call functions that implements iterations of ML methods with two techniques.
# The two techniques are random-split and cross-validation.
list_df_metrics = random_split() 
cross_validation(list_df_metrics)