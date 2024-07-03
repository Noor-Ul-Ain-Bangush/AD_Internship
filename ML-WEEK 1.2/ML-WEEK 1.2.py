# <<<--------ML-WEEK 1.2---------->>>

# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Load & Inspect dataset:
train_path = r'E:\A&D_Intership\ML-WEEK 1.2\TASK (ML-WEEK 1.2)\train.csv'
test_path = r'E:\A&D_Intership\ML-WEEK 1.2\TASK (ML-WEEK 1.2)\test.csv'
gender_submission_path = r'E:\A&D_Intership\ML-WEEK 1.2\TASK (ML-WEEK 1.2)\gender_submission.csv'

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
df_gender_submission = pd.read_csv(gender_submission_path)

# Add 'Survived' column to the test set from gender_submission
df_test['Survived'] = df_gender_submission['Survived']

# Combine train and test sets
titanic_df = pd.concat([df_train, df_test], ignore_index=True)

print(titanic_df.head())
print(titanic_df.describe())
print(titanic_df.info())
print(titanic_df.isnull().sum())

# Visualize missing values
sns.heatmap(titanic_df.isnull(), cbar=False, cmap='viridis')
plt.show()

# 2. Data Cleaning
# Handle missing values by filling them with the median or mode
titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].median())
titanic_df['Embarked'] = titanic_df['Embarked'].fillna(titanic_df['Embarked'].mode()[0])
titanic_df['Fare'] = titanic_df['Fare'].fillna(titanic_df['Fare'].median())

# Drop the 'Cabin' column as it has too many missing values
titanic_df = titanic_df.drop(columns=['Cabin'])

# Check for outliers in 'Fare' using boxplot
sns.boxplot(x=titanic_df['Fare'])
plt.show()

# Handle outliers (e.g., cap them at the 99th percentile)
fare_cap = titanic_df['Fare'].quantile(0.99)
titanic_df['Fare'] = np.where(titanic_df['Fare'] > fare_cap, fare_cap, titanic_df['Fare'])

# 3. Data Transformation
# Convert categorical data into numeric format using one-hot encoding or label encoding
categorical_features = ['Sex', 'Embarked']
numeric_features = ['Age', 'Fare']

# One-hot encode categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)])

# Apply the transformations
titanic_transformed = preprocessor.fit_transform(titanic_df)

# Convert the transformed data back to a DataFrame
transformed_df = pd.DataFrame(titanic_transformed, columns=['Age', 'Fare'] +
                              list(preprocessor.transformers_[1][1].get_feature_names_out(categorical_features)))

# Add the target column 'Survived' to the transformed DataFrame
transformed_df['Survived'] = titanic_df['Survived'].values

print(transformed_df.head())

# Save the cleaned and transformed DataFrame to a new CSV file
output_path = r'E:\A&D_Intership\ML-WEEK 1.2\TASK (ML-WEEK 1.2)\cleaned_titanic.csv'
transformed_df.to_csv(output_path, index=False)
