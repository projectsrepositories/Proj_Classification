This is a sample project for binary classification using 4 machine learning methods: 
- Logistic Regression (LR) 
- Random Forest (RF)
- K Neighbors (KN)
- Support Vector Machine (SVM)

###For the models, assign the predictor variable names as elements of the list *X_columns* in the file *params.py*.

1. To install required packages for your environment from *requirements.txt*, run the following command:   
*$ pip install -r requirements.txt*  

2. Input parameters such as number of folds(k) for CV can be changed in the *params.py*.

3. To run the binary classification project from the *src* directory:   
*$ python binary_classification.py*

4. To run the project from the jupyter notebook, execute following command in the *code cell*:   
*import binary_classification*  

Outputs of the project in jupyter notebook is available in the *classification.ipynb*.   
While running in the notebook, uncomment the first two lines in *analysis.py* for importing IPython, if necessary.

