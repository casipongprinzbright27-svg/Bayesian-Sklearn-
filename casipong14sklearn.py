"""
Group Assignment: Scikit-Learn Clustering on Patient Dataset
Block: CAS-05-601P
Date: 2026-03-14

This script demonstrates:
1. Loading unsupervised dataset (from last week's unsupervised.py)
2. Preprocessing with StandardScaler
3. Training K-Means Clustering
4. Analyzing clusters

Introduction of sklearn

Scikit-learn, or sklearn, is an open-source library for machine learning tailored for the Python programming language. It offers instruments for analysis, modeling, and forecasting of data. The library contains numerous algorithms employed in machine learning, including classification, regression, clustering, and dimensionality reduction. It is constructed upon scientific computing libraries such as NumPy and SciPy, which enhances its efficiency for managing large datasets and executing numerical operations.

Scikit-learn is popular in data science and machine learning as it provides straightforward and effective tools for predictive data analysis. It also offers a uniform programming interface that enables users to conveniently train models and generate predictions with commands like fit() and predict().

Main Characteristics of sklearn

Scikit-learn provides various key functionalities that contribute to its popularity among researchers and developers.

A main attribute is its straightforward and uniform API, enabling users to effortlessly alternate between various machine learning algorithms. The majority of models adhere to a similar procedure: training the model with fit() and producing predictions with predict().

An additional characteristic is its extensive range of machine learning tools. It offers functions that are immediately applicable for data preprocessing, model training, performance evaluation, and parameter tuning. These tools aid in decreasing development time since users aren't required to create algorithms from the beginning.

Scikit-learn works effectively with other Python libraries like NumPy, Pandas, and Matplotlib. This enables users to effortlessly handle data, conduct calculations, and visualize outcomes within a single workflow.



Algorithms Offered in sklearn

Scikit-learn offers various machine learning algorithms applicable to diverse problem types.

Available for classification tasks are algorithms like Support Vector Machines, Decision Trees, Logistic Regression, Random Forest, and Naive Bayes. These algorithms are utilized to forecast categories or labels within datasets.

For regression tasks, methods like Linear Regression and Ridge Regression are employed to forecast continuous values like prices or scores.

For clustering, methods like K-Means and DBSCAN are employed to categorize similar data points collectively without prior labels.

Furthermore, sklearn offers dimensionality reduction methods such as Principal Component Analysis (PCA) to diminish the number of variables in a dataset while maintaining crucial information

References:
http://scikit-learn.org/stable/
https://www.geeksforgeeks.org/machine-learning/learning-model-building-scikit-learn-python-machine-learning-library/

"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ============================================
# STEP 1: Load and Explore Data
# ============================================
print("=" * 60)
print("SCIKIT-LEARN: UNSUPERVISED CLUSTERING EXAMPLE")
print("=" * 60)

# Load the dataset
data = pd.read_csv('unsupervised_patient_dataset.csv')

print("\n📊 Dataset Overview:")
print(f"Shape: {data.shape} (rows, columns)")
print(f"\nFirst 5 rows:\n{data.head()}")
print(f"\nBasic Statistics:\n{data.describe()}")

# ============================================
# STEP 2: Preprocess Data with StandardScaler
# ============================================
print("\n" + "=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)

# Standardize features (important for K-Means!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

print(f"\n✅ Data standardized!")
print(f"Original data (first row): {data.iloc[0].values}")
print(f"Scaled data (first row): {X_scaled[0]}")

# ============================================
# STEP 3: Find Optimal Number of Clusters
# ============================================
print("\n" + "=" * 60)
print("FINDING OPTIMAL NUMBER OF CLUSTERS")
print("=" * 60)

silhouette_scores = []
K_range = range(2, 8)

for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_temp = kmeans_temp.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels_temp)
    silhouette_scores.append(score)
    print(f"K={k}: Silhouette Score = {score:.4f}")

# Find best k
best_k = K_range[np.argmax(silhouette_scores)]
print(f"\n🎯 Best number of clusters: {best_k}")

# ============================================
# STEP 4: Train K-Means with Optimal K
# ============================================
print("\n" + "=" * 60)
print(f"TRAINING K-MEANS CLUSTERING (K={best_k})")
print("=" * 60)

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10,
    max_iter=300
)

# Fit and predict clusters
clusters = kmeans.fit_predict(X_scaled)
data['Cluster'] = clusters

print(f"\n✅ Clustering complete!")
print(f"Inertia (within-cluster sum): {kmeans.inertia_:.4f}")

# ============================================
# STEP 5: Analyze Clusters
# ============================================
print("\n" + "=" * 60)
print("CLUSTER ANALYSIS")
print("=" * 60)

print(f"\n📊 Cluster Distribution:")
print(data['Cluster'].value_counts().sort_index())

print(f"\n📈 Cluster Centers (scaled values):\n{kmeans.cluster_centers_}")

print(f"\n📋 Cluster Statistics:")
for cluster_id in range(best_k):
    cluster_data = data[data['Cluster'] == cluster_id]
    print(f"\n--- Cluster {cluster_id} ---")
    print(f"Size: {len(cluster_data)} patients")
    print(f"Average Age: {cluster_data['Age'].mean():.1f}")
    print(f"Average Blood Pressure: {cluster_data['BloodPressure'].mean():.1f}")
    print(f"Average Cholesterol: {cluster_data['Cholesterol'].mean():.1f}")
    print(f"Average BMI: {cluster_data['BMI'].mean():.1f}")
    print(f"Average Glucose: {cluster_data['Glucose'].mean():.1f}")

# ============================================
# STEP 6: Display Results
# ============================================
print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(f"\nFirst 10 rows with cluster assignments:\n")
print(data.head(10)[['Age', 'BloodPressure', 'Cholesterol', 'BMI', 'Glucose', 'Cluster']])

print("\n" + "=" * 60)
print("✨ CLUSTERING COMPLETE!")
print("=" * 60)

