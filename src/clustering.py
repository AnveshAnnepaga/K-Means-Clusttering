from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def run_kmeans(data, k, random_state=42):
    model = KMeans(n_clusters=k, random_state=random_state)
    labels = model.fit_predict(data)
    return model, labels

def evaluate_k(data, k_range):
    wcss = []
    silhouette = []

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(data)
        wcss.append(model.inertia_)
        silhouette.append(silhouette_score(data, labels))

    return wcss, silhouette