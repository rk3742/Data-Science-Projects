import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Drop id column
df.drop('id', axis=1, inplace=True)

# Fill missing values
df['bmi'].fillna(df['bmi'].median(), inplace=True)

# Encode categorical columns
le = LabelEncoder()
cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# Features and target
X = df.drop('stroke', axis=1)
y = df['stroke']

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Accuracy
print(f"Accuracy: {accuracy_score(y_test, model.predict(X_test)) * 100:.2f}%")

# Save model
pickle.dump(model, open('model.pkl', 'wb'))
print("Model saved as model.pkl ✅")