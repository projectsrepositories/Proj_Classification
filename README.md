### Binary Classification

This is a sample project for binary classification using 4 machine learning methods: 
- Logistic Regression (LR) 
- Random Forest (RF)
- K Neighbors (KN)
- Support Vector Machine (SVM)

1. To install required packages for your environment from *requirements.txt*, run the following command:   
*$ pip install -r requirements.txt*  

2. To run the project from the jupyter notebook, 
    - First, uncomment (if commented) the following second line of code in *src/visualization.py* as below:  
      *get_ipython().run_line_magic('matplotlib', 'inline')*  

    - Second, execute the following code in the *code cell*:       
      *import binary_classification* 

3. To run the project from command line  
    - First, comment (if uncommented) the second line of code in the file *src/visualizationo.py* as below:  
      *#get_ipython().run_line_magic('matplotlib', 'inline')*  

    - Second, execute the following command:  
      *$ python binary_classification.py*     

4. Input parameters such as predictor variable names and number of folds(k) for CV  can be changed in the *params.py*.  

**Example output** of the project in jupyter notebook is available in the file ***src/classification.ipynb*.**  
**Data** output files are available in the ***output*** folder.  
**Plot** output files are available in the ***plots*** folder.  
