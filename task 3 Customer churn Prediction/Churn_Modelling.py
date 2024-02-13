import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
df = pd.read_csv('Churn_Modelling.csv')

# Preprocessing: Convert categorical variables to dummy variables and drop unnecessary columns
df_processed = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)
df_processed.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

# Split data into features (X) and target (y)
X = df_processed.drop('Exited', axis=1)
y = df_processed['Exited']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature selection using RandomForestClassifier
select = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
select.fit(X_train_scaled, y_train)

X_train_selected = select.transform(X_train_scaled)
X_test_selected = select.transform(X_test_scaled)

# Train Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_selected, y_train)

# Predictions
y_pred_train = rf_classifier.predict(X_train_selected)
y_pred_test = rf_classifier.predict(X_test_selected)

# Evaluate model
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

# Create a horizontal bar plot with color gradient
labels = ['Training Accuracy', 'Testing Accuracy']
accuracy_scores = [train_accuracy, test_accuracy]
colors = ['#1f77b4', '#ff7f0e']

plt.figure(figsize=(10, 6))
bars = plt.barh(labels, accuracy_scores, color=colors)

# Add value labels to the bars
for bar, score in zip(bars, accuracy_scores):
    plt.text(bar.get_width() - 0.02, bar.get_y() + bar.get_height()/2, f'{score:.2f}', 
             va='center', ha='right', color='white')

plt.xlabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.gca().invert_yaxis()  # Invert y-axis to have 'Training Accuracy' at the top
plt.show()

# Ask user for input
print("\nEnter customer details for prediction:")
user_input = {}
for column in X.columns:
    value = input(f"Enter the value for {column}: ")
    user_input[column] = [float(value)]

# Create a DataFrame from the user input
user_data_df = pd.DataFrame(user_input)

# Scale the user input
user_data_scaled = scaler.transform(user_data_df)
user_data_selected = select.transform(user_data_scaled)

# Predict churn for user input
churn_prediction = rf_classifier.predict(user_data_selected)

# Map prediction to human-readable format
prediction_label = "Churned" if churn_prediction[0] == 1 else "Not Churned"

print("\nPredicted Churn:", prediction_label)
