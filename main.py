import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("/content/heart_disease.csv")
df.head()

sns.pairplot(df[['age']])
plt.show()
sns.pairplot(df[['target']])
plt.show()

plt.figure(figsize=(15,7))
sns.heatmap(df.corr(),annot=True,cmap='coolwarm',vmin=-1,vmax=1)
plt.title("Feature Correlation Heatmap")
plt.show()

x = df[['age','cp','thalach']]
y = df['target']

#split Data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# model initialization and training
model = LogisticRegression()
model.fit(x_train, y_train)

#Predictions and performance metrics
y_pred = model.predict(x_test)
y_proba = model.predict_proba(x_test)

from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, accuracy_score
# model performance
arruracy = accuracy_score(y_test,y_pred)
print(f"Accuracy: {arruracy:.2f}")

class_pred = classification_report(y_test,y_pred)

def max_heart_rate():
    try:
        age = int(input("Enter Age (15 years or older): "))
        if age < 15:
            print("Error: Age must be 15 years or older.")
            return
        
        cp = int(input("Enter Chest Pain Type (0-3): "))
        if cp not in [0, 1, 2, 3]:
            print("Error: Chest Pain Type must be between 0 and 3.")
            return
        
        thalach = int(input("Enter Maximum Heart Rate Achieved (30 bpm or higher): "))
        if thalach < 30:
            print("Error: Maximum Heart Rate must be 30 bpm or higher.")
            return

        # Create DataFrame
        user_data = pd.DataFrame([[age, cp, thalach]], columns=['age', 'cp', 'thalach'])

        # Predict
        predict = model.predict(user_data)
        if predict == 0:
            print("No, Heart Disease.")
        else:
            print("Yes, the person is suffering from Heart Disease.")
    except ValueError:
        print("Invalid input! Please enter numeric values.")

# Call the function
max_heart_rate()
