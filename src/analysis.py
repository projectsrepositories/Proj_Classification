#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.metrics import recall_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import math

def plot_confusion_matrix(y_test, yhats, methods, labels, num_run):
    # Plot the confusion-matrix for the last iteration of the random_split predictions.
    nrows = math.ceil(len(methods)/2)
    ncols = 2
    row_idx = 0
    col_idx = 0
    fig, ax= plt.subplots(nrows = nrows, ncols = ncols)
    index = 0
    for row in ax:
        for ax_cur in row:                      
            if index<len(methods):
                cm = confusion_matrix(y_test, yhats[index])
                sns.heatmap(cm, annot=True, ax = ax_cur); 
                ax_cur.set_xlabel('Predicted labels')
                ax_cur.set_ylabel('True labels')
                ax_cur.set_title(methods[index])
                ax_cur.set_xticklabels(labels)
                ax_cur.set_yticklabels(labels)                
            index+=1        
    plt.tight_layout()
    plt.subplots_adjust(hspace=.9, top=0.9)
    plt.suptitle(f'Confusion Matrix for the last of the {num_run} iterations', fontweight ="bold")
    plt.savefig('../plots/Confusion_matrix_for_last_run.png',bbox_inches='tight')
    plt.show()

def plot_boxplot(list_df, titles, num_run):
    # Create 4 boxplots (accuracy, auc, recall, precision) in the same file.
    # Each boxplot is for the total iterations results for the random_split predictions.
    fig, ax = plt.subplots(nrows=2, ncols=2)
    index = 0    
    for row in ax:
        for ax_cur in row:
            list_df[index].plot(kind='box', ax=ax_cur)
            ax_cur.set_title(titles[index])
            index+=1
    plt.tight_layout()
    plt.subplots_adjust(hspace=.9, top=0.9)
    plt.suptitle(f'Randomly selecting Test sets in each of the {num_run} iterations', fontweight ="bold")
    plt.savefig(f'../plots/Metrics_{num_run}_box.png',bbox_inches='tight')
    plt.show()    

def plot_bar_charts(list_df_metrics, titles):
    # Create 4 bar charts (accuracy, auc, recall and precision) in the same file.
    # Each bar chart compares the random_split results against cross-validation results.
    fig, ax = plt.subplots(nrows=2, ncols=2)
    index = 0
    for row in ax:
        for ax_cur in row:
            ymin = list_df_metrics[index].values.min()
            ymax = list_df_metrics[index].values.max()

            list_df_metrics[index].plot(kind='bar', ax = ax_cur)  
            ax_cur.set_title(titles[index])            
            ax_cur.set_ylim(bottom=ymin-0.05,top=ymax+0.1)
            ax_cur.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)
            index+=1   
    plt.tight_layout()
    plt.subplots_adjust(hspace=.9, top=0.9)
    plt.suptitle('Cross-validation (CV) vs Randomly selecting Test sets', fontweight ="bold")
    cv = list_df_metrics[0].columns[1]
    num_run = list_df_metrics[0].columns[0]
    plt.savefig(f'../plots/Metrics_{cv}_{num_run}_bar.png',bbox_inches='tight')
    plt.show()     

def evaluate_models(y_test, yhat, yhat_prob):
    # Calculate accuracy, auc, recall, and precision
    metrics = dict()
    metrics['acc'] = accuracy_score(y_test, yhat)
    metrics['auc'] = roc_auc_score(y_test, yhat_prob[:,1])
    metrics['rec'] = recall_score(y_test, yhat)
    metrics['pre'] = precision_score(y_test, yhat)
    return metrics

