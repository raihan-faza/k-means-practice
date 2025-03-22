import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# print basic info
df = pd.read_csv("penguins.csv")
print(df.info())
print(df.describe())
print(df.isnull().sum())

# removing null data
df.dropna(inplace=True)
print(df.isnull().sum())

# locating data
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# training the model
# Try different k values and compute silhouette score
sil_scores = []

for k in range(2, 10):  # k must be at least 2
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    sil_scores.append(silhouette_score(X, labels))

# Plot silhouette scores
plt.plot(range(2, 10), sil_scores, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for Optimal k")
plt.show()
