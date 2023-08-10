"""Input parameters for the binary classification."""

filename_ip_data = "../inputfile/ChurnData.csv"
X_columns = ['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']
y_column = 'churn'
labels = ['No Churn', 'Churn']
num_run = 10  # Number of run for random_split
cv = 5  # Number of fold for cross-validation
test_size = 0.2  # Test size for train_test_split
seed = 0  # Random seed for reproducibility