# Loan-predction-Project
!pip install squarify
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
import squarify

train = pd.read_csv(r'C:\Users\91970\Downloads\loan_approval_dataset.csv')
train.head()

test = pd.read_csv(r'C:\Users\91970\Downloads\loan_approval_dataset.csv')
test.head()

train_original=train.copy()
test_original=test.copy()

train.columns
test.columns
train.dtypes
train.shape
test.shape
train['loan_status'].value_counts()
train['loan_status'].value_counts(normalize=True)
train['loan_status'].value_counts().plot.bar()
train['education'].value_counts(normalize=True).plot.bar(title='education')
plt.show()
train['self_employed'].value_counts(normalize=True).plot.bar(title='self_employed')
plt.show()
train['income_annum'].value_counts(normalize=True).plot(kind='hist', bins=63, title='Normalized Histogram of Annual Income')
plt.xlabel('Proportion')
plt.ylabel('Frequency')
plt.show()
train['loan_amount'].value_counts(normalize=True).plot(kind='hist', bins=63, title='Loan Amount')
plt.xlabel('Loan Amount')
plt.ylabel('Frequency')
plt.show()
train['loan_term'].value_counts(normalize=True).plot.bar(title='loan_term')
plt.show()
train['cibil_score'].value_counts(normalize=True).plot(kind='hist', bins=63, title='CIBIL SCORE')
plt.xlabel('CIBIL Score')
plt.ylabel('Frequency')
plt.show()

dependents_count = train['no_of_dependents'].value_counts(normalize=True)
# Creating the pie chart
plt.figure(figsize=(8, 8))  # Set the figure size for better visibility
plt.pie(dependents_count, labels=dependents_count.index, autopct='%1.1f%%', startangle=140)
plt.title('Pie Chart of Dependents')  # Adding a title to the pie chart
plt.show() 

plt.figure(figsize=(10, 6))  
train['residential_assets_value'].plot.box()  
plt.title('Box Plot of Residential Assets Value')  
plt.ylabel('Value')  # Setting the y-axis label
plt.grid(True)  # Adding a grid for easier reading of values
plt.show() 

plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
sns.violinplot(data=train, y='commercial_assets_value')  # y for vertical plot
plt.title('Violin Plot of Commercial Assets Value')  # Adding a title
plt.ylabel('Commercial Assets Value')  # Labeling the y-axis
plt.show()

plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
train['bank_asset_value'].hist(bins=30)  # You can adjust the number of bins to better fit your data distribution
plt.title('Histogram of Bank Assets Value')  # Adding a title
plt.xlabel('Bank Assets Value')  # Labeling the x-axis
plt.ylabel('Frequency')  # Labeling the y-axis
plt.show() 

loan_status_counts = train['loan_status'].value_counts(normalize=True)
labels = [f'{status}\n{proportion:.2%}' for status, proportion in loan_status_counts.items()]

# Creating the tree map
plt.figure(figsize=(12, 6))
squarify.plot(sizes=loan_status_counts, label=labels, alpha=0.8)
plt.axis('off')  # Removes the axes
plt.title('Tree Map of Loan Status Proportions')
plt.show()
education=pd.crosstab(train['education'],train['loan_status'])
education.div(education.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.show()
no_of_dependents=pd.crosstab(train['no_of_dependents'],train['loan_status'])

self_employed=pd.crosstab(train['self_employed'],train['loan_status'])
loan_term=pd.crosstab(train['loan_term'],train['loan_status'])


no_of_dependents.div(no_of_dependents.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.show()

self_employed.div(self_employed.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.show()
loan_term.div(loan_term.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.show()
train.groupby('loan_status')['income_annum'].mean().plot.bar()
bins = [0, 300, 500, 600, 700, 850]  # Adjust as necessary for your data
labels = ['Poor', 'Fair', 'Good', 'Very Good', 'Exceptional']
train['cibil_score_bin'] = pd.cut(train['cibil_score'], bins, labels=labels)
cibil_score_bin = pd.crosstab(train['cibil_score_bin'], train['loan_status'])
cibil_score_bin.div(cibil_score_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('cibil_score')
P = plt.ylabel('Percentage')
bins = [1000000, 5000000, 10000000, 50000000]  # Adjust as necessary for your data
labels = ['Good', 'Very Good', 'Exceptional']
train['residential_assets_value_bin'] = pd.cut(train['residential_assets_value'], bins, labels=labels)
residential_assets_value_bin = pd.crosstab(train['residential_assets_value_bin'], train['loan_status'])
residential_assets_value_bin.div(residential_assets_value_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('residential_assets_value')
P = plt.ylabel('Percentage')
bins = [0, 1000000, 5000000, 10000000, 50000000]  # Adjust as necessary for your data
labels = ['Poor', 'Good', 'Very Good', 'Exceptional']

# Apply pd.cut()
train['commercial_assets_value_bin'] = pd.cut(train['commercial_assets_value'], bins, labels=labels)

# Create crosstab and plot
commercial_assets_value_bin = pd.crosstab(train['commercial_assets_value_bin'], train['loan_status'])
commercial_assets_value_bin.div(commercial_assets_value_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('Commercial Assets Value')
plt.ylabel('Percentage')
train.isnull().sum()
train['no_of_dependents'].fillna(train['no_of_dependents'].mode()[0], inplace=True)
train['education'].fillna(train['education'].mode()[0], inplace=True)
train['self_employed'].fillna(train['self_employed'].mode()[0], inplace=True)
train['income_annum'].fillna(train['income_annum'].mode()[0], inplace=True)
train['loan_status'].fillna(train['loan_status'].mode()[0], inplace=True)
train['loan_amount'].fillna(train['loan_amount'].mode()[0], inplace=True)
train['loan_term'].fillna(train['loan_term'].mode()[0], inplace=True)
train['cibil_score'].fillna(train['cibil_score'].mode()[0], inplace=True)
train['residential_assets_value'].fillna(train['residential_assets_value'].mode()[0], inplace=True)
train['commercial_assets_value'].fillna(train['commercial_assets_value'].mode()[0], inplace=True)
train['luxury_assets_value'].fillna(train['luxury_assets_value'].mode()[0], inplace=True)
train['bank_asset_value'].fillna(train['bank_asset_value'].mode()[0], inplace=True)
train['loan_id'].fillna(train['loan_id'].mode()[0], inplace=True)
train['loan_id'].value_counts()
train.isnull().sum()
test['loan_id'].fillna(train['loan_id'].mode()[0], inplace=True)
test['no_of_dependents'].fillna(train['no_of_dependents'].mode()[0], inplace=True)
test['education'].fillna(train['education'].mode()[0], inplace=True)
test['self_employed'].fillna(train['self_employed'].mode()[0], inplace=True)
test['income_annum'].fillna(train['income_annum'].mode()[0], inplace=True)
test['loan_amount'].fillna(train['loan_amount'].mode()[0], inplace=True)
test['loan_term'].fillna(train['loan_term'].median(), inplace=True)
test['loan_status'].fillna(train['loan_status'].mode()[0], inplace=True)  
train['LoanAmount_log']=np.log(train['loan_amount'])
train['LoanAmount_log'].hist(bins=20)
test['LoanAmount_log']=np.log(test['loan_amount'])
# Dropping 'loan_id' column if it exists
if 'loan_id' in train.columns:
    train = train.drop('loan_id', axis=1)
else:
    print("Column 'loan_id' not found in train dataframe.")

if 'loan_id' in test.columns:
    test = test.drop('loan_id', axis=1)
else:
    print("Column 'loan_id' not found in test dataframe.")

# Dropping 'loan_status' from training dataset and storing it as target variable
if 'loan_status' in train.columns:
    y = train['loan_status']
    train = train.drop('loan_status', axis=1)
else:
    print("Column 'loan_status' not found in train dataframe.")

# Apply pd.get_dummies to handle categorical variables
train = pd.get_dummies(train)
test = pd.get_dummies(test)

# Aligning the columns of the test dataset to match the training dataset
# Ensure test set has all the columns that model expects
test = test.reindex(columns=train.columns, fill_value=0)

# Splitting the dataset into training and validation parts
from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(train, y, test_size=0.3, random_state=42)

# Logistic Regression model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model = LogisticRegression(max_iter=1000)  # Increase max_iter if convergence issues occur
model.fit(x_train, y_train)

# Predicting on validation set
pred_cv = model.predict(x_cv)
print("Validation Accuracy:", accuracy_score(y_cv, pred_cv))

# Predicting on test set
pred_test = model.predict(test)
sample_submission_path = r'C:\Users\91970\Downloads\sample_submission_49d68Cx.csv'
sample_submission = pd.read_csv(sample_submission_path)

# Display the first few rows to check the format
print(sample_submission.head())
submission['Loan_Status']=pred_test
submission['Loan_ID']=test_original['loan_id']
print(submission)
submission['Loan_Status'].replace(0, 'N', inplace=True)
submission['Loan_Status'].replace(1, 'Y', inplace=True)
print(submission)
pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv(r'C:\Users\91970\Downloads\sample_submission_49d68Cx.csv')
rom sklearn.model_selection import StratifiedKFold
i = 1
mean = 0
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)  # Set shuffle to True

for train_index, test_index in kf.split(X, y):
    print('\n{} of kfold {}'.format(i, kf.n_splits))
    xtr, xvl = X.iloc[train_index], X.iloc[test_index]
    ytr, yvl = y[train_index], y[test_index]
    
    model = LogisticRegression(random_state=1)
    model.fit(xtr, ytr)
    
    pred_test = model.predict(xvl)
    score = accuracy_score(yvl, pred_test)
    mean += score
    print('accuracy_score:', score)
    
    i += 1

print('\nMean Validation Accuracy:', mean / (i - 1))
from sklearn import metrics
fpr, tpr, _ = metrics.roc_curve(yvl, pred)
auc = metrics.roc_auc_score(yvl, pred)
plt.figure(figsize=(12,8))
plt.plot(fpr, tpr, label="validation, auc="+str(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
plt.show()
submission['Loan_Status'].replace(0, 'N', inplace=True)
submission['Loan_Status'].replace(1, 'Y', inplace=True)
pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv(r'C:\Users\91970\Downloads\Outputlogistic_dataset.csv')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np

# Assuming X and y are your feature matrix and target vector, respectively

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)  # Set shuffle to True
mean = 0
i = 1

for train_index, test_index in kf.split(X, y):
    print(f'\n{i} of kfold {kf.n_splits}')
    xtr, xvl = X.iloc[train_index], X.iloc[test_index]
    ytr, yvl = y.iloc[train_index], y.iloc[test_index]
    
    # Use RandomForestClassifier instead of LogisticRegression
    model = RandomForestClassifier(random_state=1, max_depth=10)  # Set max_depth to control complexity
    model.fit(xtr, ytr)
    
    pred_vl = model.predict(xvl)
    score = accuracy_score(yvl, pred_vl)
    mean += score
    print('accuracy_score:', score)
    
    i += 1

mean_accuracy = mean / (i - 1)
print(f'\nMean Validation Accuracy: {mean_accuracy}')
pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv(r'C:\Users\91970\Downloads\Oplogistic_dataset.csv')

