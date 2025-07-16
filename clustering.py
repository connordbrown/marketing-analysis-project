import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import datetime
import seaborn as sns

# Read the cleaned data
df = pd.read_csv('online_retail_II_cleaned.csv')

# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Calculate RFM metrics
snapshot_date = df['InvoiceDate'].max() + datetime.timedelta(days=1)

rfm = df.groupby('Customer ID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
    'Invoice': 'count',  # Frequency
    'TotalValue': 'sum'  # Monetary
}).rename(columns={
    'InvoiceDate': 'Recency',
    'Invoice': 'Frequency',
    'TotalValue': 'Monetary'
})

# Remove any missing values
rfm = rfm.dropna()

# Scale the RFM metrics
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)
rfm_scaled = pd.DataFrame(rfm_scaled, columns=['Recency', 'Frequency', 'Monetary'], index=rfm.index)

# Elbow Method
wcss = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(rfm_scaled)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(rfm_scaled, kmeans.labels_))

# Plot WCSS (Elbow Method)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(k_range, wcss, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.title('Elbow Method')

# Plot Silhouette Scores
plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis')

plt.tight_layout()
plt.savefig('clustering_analysis.png')
plt.close()

# Choose optimal k based on analysis (example: k=4)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Analyze clusters
cluster_analysis = rfm.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean'
}).round(2)

# Save results
cluster_analysis.to_csv('cluster_analysis.csv')
rfm.to_csv('customer_segments.csv')

# Print summary
print("\nCluster Analysis Summary:")
print(cluster_analysis)

# Visualize clusters
plt.figure(figsize=(12, 8))
scatter = plt.scatter(rfm_scaled['Recency'], 
                     rfm_scaled['Frequency'],
                     c=rfm['Cluster'],
                     cmap='viridis')
plt.xlabel('Recency (Standardized)')
plt.ylabel('Frequency (Standardized)')
plt.title('Customer Segments')
plt.colorbar(scatter)
plt.savefig('customer_segments.png')
plt.close()

# Print additional insights
print("\nCustomer Segment Sizes:")
print(rfm['Cluster'].value_counts().sort_index())

print("\nOptimal number of clusters based on Silhouette Score:", 
      k_range[np.argmax(silhouette_scores)])

# Print cluster characteristics
print("\nCluster Profiles:")
for cluster in range(optimal_k):
    cluster_data = rfm[rfm['Cluster'] == cluster]
    print(f"\nCluster {cluster} ({len(cluster_data)} customers):")
    print(f"Recency: {cluster_data['Recency'].mean():.1f} days (mean)")
    print(f"Frequency: {cluster_data['Frequency'].mean():.1f} purchases (mean)")
    print(f"Monetary: ${cluster_data['Monetary'].mean():.2f} (mean)")
print()