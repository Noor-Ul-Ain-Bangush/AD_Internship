import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the datasets
data1 = pd.read_csv(r'E:\A&D_Intership\ML-WEEK 2.2\test.csv')
data2 = pd.read_csv(r'E:\A&D_Intership\ML-WEEK 2.2\train.csv')

# Combine the datasets if needed, assuming you need to merge them
data = pd.concat([data1, data2], ignore_index=True)

# Preprocessing
# Fill missing values with the most frequent value for each column
imputer = SimpleImputer(strategy='most_frequent')
data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Encode categorical variables
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Assuming 'Loan_Status' is the target variable
# Split data into features and target variable
X = data.drop(columns=['Loan_Status'])
y = data['Loan_Status']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Train Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
dt_predictions = dt_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)

# Evaluate models
dt_accuracy = accuracy_score(y_test, dt_predictions)
rf_accuracy = accuracy_score(y_test, rf_predictions)

print(f'Decision Tree Accuracy: {dt_accuracy}')
print(f'Random Forest Accuracy: {rf_accuracy}')

print('\nDecision Tree Classification Report:')
print(classification_report(y_test, dt_predictions))

print('\nRandom Forest Classification Report:')
print(classification_report(y_test, rf_predictions))
