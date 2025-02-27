import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


if os.path.exists("/kaggle/input/iris/Iris.csv"):
    df = pd.read_csv("/kaggle/input/iris/Iris.csv")
else:
    df = pd.read_csv(r"C:\Users\User\Downloads\Documents\Iris.csv")


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(df.drop('Species', axis=1))
scaled_feature = scaler.transform(df.drop('Species', axis=1))
df_feat = pd.DataFrame(scaled_feature, columns=df.columns[:-1])


from sklearn.model_selection import train_test_split
X = df_feat
y = df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, pred)
print(f"Accuracy: {accuracy * 100:.2f}%")


submission = pd.DataFrame({"Id": X_test.index, "Species": pred})
submission.to_csv("submission.csv", index=False)

print("Submission file saved as submission.csv")
