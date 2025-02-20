# Heart Disease Prediction

## Overview

This project analyzes heart disease data and builds a logistic regression model to predict the likelihood of heart disease. The dataset includes various health-related attributes such as age, cholesterol levels, and chest pain type.

## Dataset

The dataset used for this project is `heart_disease_data.csv`, which contains the following columns:

- **age**: Age of the patient.
- **sex**: Gender (1 = male, 0 = female).
- **cp**: Chest pain type (1-4).
- **trestbps**: Resting blood pressure.
- **chol**: Serum cholesterol level.
- **fbs**: Fasting blood sugar (>120 mg/dl, 1 = true, 0 = false).
- **restecg**: Resting electrocardiographic results.
- **thalach**: Maximum heart rate achieved.
- **exang**: Exercise-induced angina (1 = yes, 0 = no).
- **oldpeak**: ST depression induced by exercise.
- **slope**: Slope of the peak exercise ST segment.
- **ca**: Number of major vessels colored by fluoroscopy.
- **thal**: Thalassemia defect type.
- **target**: Presence of heart disease (1 = disease, 0 = no disease).

## Installation

To run this project, install the required dependencies:

```sh
pip install pandas numpy seaborn matplotlib scikit-learn
```

## Usage

### Load the dataset:

```python
import pandas as pd
df = pd.read_csv("heart_disease_data.csv")
```

### Perform exploratory data analysis:

```python
import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(x='target', data=df)
plt.show()
```

### Train the logistic regression model:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```

### Save the model using pickle:

```python
import pickle
with open('heart_disease_model.pkl', 'wb') as file:
    pickle.dump(model, file)
```

## Results

- The model is trained using logistic regression.
- It predicts whether a patient has heart disease based on medical attributes.
- The accuracy of the model can be evaluated using accuracy score metrics.

## License

This project is open-source and available for educational purposes.

## Author

Ahmed