#           <<<<--------ML-WEEk: 3.2--------->>>>
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 1: Generate the synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10,
                           n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)

# Convert to DataFrame for easier manipulation
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
df['target'] = y

# Step 2: Feature Creation
# Create polynomial features (interaction terms)
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
poly_features = poly.fit_transform(df.drop('target', axis=1))
poly_feature_names = poly.get_feature_names_out(df.columns[:-1])

# Convert to DataFrame
df_poly = pd.DataFrame(poly_features, columns=poly_feature_names)
df_poly['target'] = y

# Step 3: Feature Selection

# 3.1: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_poly.drop('target', axis=1), df_poly['target'], test_size=0.3, random_state=1)

# 3.2: Scale the data to be non-negative
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3.3: SelectKBest using chi2
selector_kbest = SelectKBest(score_func=chi2, k=20)
X_train_kbest = selector_kbest.fit_transform(X_train_scaled, y_train)
X_test_kbest = selector_kbest.transform(X_test_scaled)

# 3.4: Recursive Feature Elimination (RFE) with RandomForestClassifier
model_rfe = RandomForestClassifier(random_state=1)
rfe = RFE(model_rfe, n_features_to_select=20)
X_train_rfe = rfe.fit_transform(X_train, y_train)
X_test_rfe = rfe.transform(X_test)

# Step 4: Model Building

# 4.1: Baseline model with all features
model_baseline = RandomForestClassifier(random_state=1)
model_baseline.fit(X_train, y_train)
y_pred_baseline = model_baseline.predict(X_test)
baseline_accuracy = accuracy_score(y_test, y_pred_baseline)
print(f'Baseline Model Accuracy with All Features: {baseline_accuracy:.4f}')

# 4.2: Model with selected features (SelectKBest)
model_kbest = RandomForestClassifier(random_state=1)
model_kbest.fit(X_train_kbest, y_train)
y_pred_kbest = model_kbest.predict(X_test_kbest)
kbest_accuracy = accuracy_score(y_test, y_pred_kbest)
print(f'Model Accuracy with SelectKBest Features: {kbest_accuracy:.4f}')

# 4.3: Model with selected features (RFE)
model_rfe = RandomForestClassifier(random_state=1)
model_rfe.fit(X_train_rfe, y_train)
y_pred_rfe = model_rfe.predict(X_test_rfe)
rfe_accuracy = accuracy_score(y_test, y_pred_rfe)
print(f'Model Accuracy with RFE Features: {rfe_accuracy:.4f}')
