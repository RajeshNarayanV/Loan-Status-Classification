import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import joblib
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('train.csv')

# Data Preprocessing
data = data.drop('Loan_ID', axis=1)

columns = ['Gender', 'Dependents', 'LoanAmount', 'Loan_Amount_Term']
data = data.dropna(subset=columns)

data['Self_Employed'] = data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])
data['Credit_History'] = data['Credit_History'].fillna(data['Credit_History'].mode()[0])

data['Dependents'] = data['Dependents'].replace(to_replace="3+", value='4')

data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0}).astype('int')
data['Married'] = data['Married'].map({'Yes': 1, 'No': 0}).astype('int')
data['Education'] = data['Education'].map({'Graduate': 1, 'Not Graduate': 0}).astype('int')
data['Self_Employed'] = data['Self_Employed'].map({'Yes': 1, 'No': 0}).astype('int')
data['Property_Area'] = data['Property_Area'].map({'Rural': 0, 'Semiurban': 2, 'Urban': 1}).astype('int')
data['Loan_Status'] = data['Loan_Status'].map({'Y': 1, 'N': 0}).astype('int')

print(data)

X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']

cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
scaler = StandardScaler()
X[cols] = scaler.fit_transform(X[cols])

# Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Decision Tree Classifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)


# Random Forest Classifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)


# Hyperparameter Tuning for RandomForestClassifier
# defining a dictionary containing hyperparameters for the random forest classifiers
rf_grid = {
    'n_estimators': np.arange(10, 1000, 10),
    'max_features': ['auto', 'sqrt'],
    'max_depth': [None, 3, 5, 10, 20, 30],
    'min_samples_split': [2, 5, 20, 50, 100],
    'min_samples_leaf': [1, 2, 5, 10]
}

#creates an instance to search for the best hyperparameters

rs_rf = RandomizedSearchCV(RandomForestClassifier(),
                           param_distributions=rf_grid,
                           cv=5,
                           n_iter=20,
                           verbose=True)

rs_rf.fit(X, y)


# Save the trained model and scaler
joblib.dump(rf, 'loan_status_classification_model.pkl')
joblib.dump(lr, 'logistic_regression_model.pkl')
joblib.dump(dt, 'decision_tree_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Model Comparison Plot
models = ['Logistic Regression', 'Decision Tree', 'Random Forest']
accuracies = [lr.score(X_test, y_test), dt.score(X_test, y_test), rs_rf.best_score_]

plt.bar(models, accuracies)
plt.title('Performance Comparison')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('performance_comparison.png')

print("Logistic Regression score:", lr.score(X_test, y_test))
print("Decision Tree score:", dt.score(X_test, y_test))
print("RandomForestClassifier score before Hyperparameter Tuning:", rf.score(X_test, y_test))
print("RandomForestClassifier score after Hyperparameter Tuning:", rs_rf.best_score_)

#feature importance
features=X.columns.values
regressionTree_imp=rf.feature_importances_
plt.figure(figsize=(10,4))
plt.yscale('log')
plt.bar(range(len(regressionTree_imp)),regressionTree_imp,align='center')
plt.xticks(range(len(regressionTree_imp)),features,rotation='vertical')
plt.title('feature importance')
plt.ylabel('importance')
plt.show()