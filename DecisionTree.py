import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv(r"C:\Users\User\Downloads\Documents\Breast_Cancer.csv")

# Split features and labels
X = df.drop('Status', axis=1)
y = df['Status']

# Convert categorical variables to numerical
X = pd.get_dummies(X)  # One-hot encoding

# Alternatively, use Label Encoding if needed
# encoder = LabelEncoder()
# for column in X.select_dtypes(include=['object']).columns:
#     X[column] = encoder.fit_transform(X[column])

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Decision Tree model
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

# Make predictions
prediction = dtree.predict(X_test)

# Print evaluation metrics
print(confusion_matrix(y_test, prediction))
print('\n')
print(classification_report(y_test, prediction))

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)

print(confusion_matrix(y_test, rfc_pred))
print('\n')
print(classification_report(y_test, rfc_pred))

