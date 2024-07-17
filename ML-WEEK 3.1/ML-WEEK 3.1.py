import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score, adjusted_mutual_info_score
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message="When grouping with a length-1 list-like")

# Load the dataset
df = pd.read_csv(r'E:\A&D_Intership\ML-WEEK 3.1\pythonProject1\mall_customers.csv')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Display the last few rows of the dataset
print("\nLast few rows of the dataset:")
print(df.tail())

# Display the shape of the dataset
print("\nShape of the dataset:")
print(df.shape)

# Display the information about the dataset
print("\nInformation about the dataset:")
print(df.info())

# Display the statistical summary of the dataset
print("\nStatistical summary of the dataset:")
print(df.describe())


# Data Exploration
print("\nData Exploration")

fig = px.histogram(df, x='Age', nbins=10, title='Distribution of Age')
fig.show()

fig = px.histogram(df, x='Annual Income (k$)', nbins=10, title='Distribution of Annual Income')
fig.show()

fig = px.histogram(df, x='Spending Score (1-100)', nbins=10, title='Distribution of Spending Score')
fig.show()

fig = px.scatter(df, x='Age', y='Annual Income (k$)', color='Gender', hover_data=['CustomerID'])
fig.update_layout(title='Age vs Annual Income')
fig.show()

fig = px.scatter(df, x='Annual Income (k$)', y='Spending Score (1-100)', color='Gender', hover_data=['CustomerID'])
fig.update_layout(title='Annual Income vs Spending Score')
fig.show()

fig = px.box(df, x='Gender', y='Annual Income (k$)', points='all', title='Annual Income by Gender')
fig.show()

fig = px.box(df, x='Gender', y='Spending Score (1-100)', points='all', title='Spending Score by Gender')
fig.show()

# Check for missing values
print("\nChecking for missing values:")
print(df.isnull().sum())

# Encode the 'Gender' column
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])

# Select features for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Apply K-means algorithm
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# Add the cluster labels to the dataset
df['Cluster'] = y_kmeans

# Visualize the clusters
plt.figure(figsize=(10, 7))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=df, palette='viridis', s=100, alpha=0.7)
plt.title('Clusters of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# Evaluate the clustering performance
print("\nClustering Performance Evaluation")

inertia = kmeans.inertia_
print(f"Inertia (WCSS): {inertia}")

silhouette_avg = silhouette_score(X_scaled, y_kmeans)
print(f"Silhouette Score: {silhouette_avg}")

davies_bouldin = davies_bouldin_score(X_scaled, y_kmeans)
print(f"Davies-Bouldin Index: {davies_bouldin}")

calinski_harabasz = calinski_harabasz_score(X_scaled, y_kmeans)
print(f"Calinski-Harabasz Index: {calinski_harabasz}")

# Analyze and profile the characteristics of each cluster
print("\nCluster Analysis")

for cluster in df['Cluster'].unique():
    print(f"\nCluster {cluster} Characteristics:")
    print(df[df['Cluster'] == cluster].describe())
