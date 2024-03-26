import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

# Importing dataset
dataset = pd.read_csv('diabetes.csv')

# Replacing zero values with NaN
dataset[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = dataset[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].replace(0, np.NaN) 

# Replacing NaN with mean values
dataset["Glucose"].fillna(dataset["Glucose"].mean(), inplace = True)
dataset["BloodPressure"].fillna(dataset["BloodPressure"].mean(), inplace = True)
dataset["SkinThickness"].fillna(dataset["SkinThickness"].mean(), inplace = True)
dataset["Insulin"].fillna(dataset["Insulin"].mean(), inplace = True)
dataset["BMI"].fillna(dataset["BMI"].mean(), inplace = True)

# Feature scaling using MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = sc.fit_transform(dataset)
dataset_scaled = pd.DataFrame(dataset_scaled)

# Selecting features - [Glucose, Insulin, BMI, Age]
X = dataset_scaled.iloc[:, [1, 4, 5, 7]].values
Y = dataset_scaled.iloc[:, 8].values

# Splitting X and Y
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42, stratify=dataset['Outcome'])

# K nearest neighbors Algorithm
knn = KNeighborsClassifier(n_neighbors=24, metric='minkowski', p=2)
knn.fit(X_train, Y_train)

# Making predictions on test dataset
Y_pred_knn = knn.predict(X_test)

# Evaluating using accuracy_score metric
accuracy_knn = accuracy_score(Y_test, Y_pred_knn)
print("K Nearest neighbors: " + str(accuracy_knn * 100))

# Saving the trained model to a .pkl file
joblib.dump(knn, 'diabetes_model.pkl')
