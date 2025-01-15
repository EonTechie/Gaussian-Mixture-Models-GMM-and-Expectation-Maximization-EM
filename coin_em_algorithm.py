# Github : EonTechie

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

###### Step 1: Generate Random Dataset ######

# Set random seed for reproducibility
np.random.seed(42)

# Generate random centers for 3 clusters
num_clusters = 3
cluster_centers = np.random.uniform(-5, 15, size=(num_clusters, 2))  # Random means in range [-5, 15]

# Generate random isotropic covariance matrices (scale factors)
covariances = [np.eye(2) * np.random.uniform(1.0, 3.0) for _ in range(num_clusters)]

# Random sample sizes between 200 and 400
samples_per_cluster = [np.random.randint(200, 401) for _ in range(num_clusters)]

# Generate data for each cluster
data = []
for mean, cov, n in zip(cluster_centers, covariances, samples_per_cluster):
    cluster_data = np.random.multivariate_normal(mean, cov, n)
    data.append(cluster_data)
data = np.vstack(data)

# Visualize the generated data
plt.scatter(data[:, 0], data[:, 1], s=5)
plt.title("Generated Data (Random Clusters and Covariances)")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

###### Step 2: Multivariate Gaussian PDF ######

def multivariate_gaussian(x, mean, cov):
    """Compute Multivariate Gaussian PDF."""
    d = mean.shape[0]
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    diff = x - mean
    exponent = -0.5 * np.sum(diff @ inv * diff, axis=1)
    return np.exp(exponent) / (np.sqrt((2 * np.pi) ** d * det))

###### Step 3: EM Algorithm ######

def initialize_parameters(data, k):
    """Initialize means, covariances, and weights randomly."""
    n_samples, n_features = data.shape
    
    # Randomly select data points as initial means
    means = data[np.random.choice(n_samples, k, replace=False)]
    
    # Random isotropic covariance matrices
    covariances = [np.eye(n_features) * np.random.uniform(1.0, 3.0) for _ in range(k)]
    
    # Random weights normalized to sum to 1
    weights = np.random.rand(k)
    weights /= weights.sum()
    
    return means, covariances, weights

def e_step(data, means, covariances, weights, k):
    """E-Step: Compute E_probabilities."""
    n_samples = len(data)
    E_probabilities = np.zeros((n_samples, k))
    for i in range(k):
        E_probabilities[:, i] = weights[i] * multivariate_gaussian(data, means[i], covariances[i])
    E_probabilities /= E_probabilities.sum(axis=1, keepdims=True)
    return E_probabilities

def m_step(data, E_probabilities, k):
    """M-Step: Update parameters."""
    n_samples, n_features = data.shape
    new_means = np.zeros((k, n_features))
    new_covariances = []
    new_weights = np.zeros(k)

    for i in range(k):
        resp_sum = E_probabilities[:, i].sum()
        new_means[i] = (E_probabilities[:, i][:, None] * data).sum(axis=0) / resp_sum
        diff = data - new_means[i]
        cov = (E_probabilities[:, i][:, None] * diff).T @ diff / resp_sum
        new_covariances.append(np.eye(n_features) * np.mean(np.diag(cov)))  # Enforce isotropic covariance
        new_weights[i] = resp_sum / n_samples
    
    return new_means, new_covariances, new_weights

def em_algorithm(data, k, max_iter=100, tol=1e-4):
    """EM Algorithm for Gaussian Mixture Models."""
    means, covariances, weights = initialize_parameters(data, k)
    
    for iteration in range(max_iter):
        # E-Step
        E_probabilities = e_step(data, means, covariances, weights, k)
        
        # M-Step
        new_means, new_covariances, new_weights = m_step(data, E_probabilities, k)
        
        # Convergence check
        if np.linalg.norm(new_means - means) < tol:
            print(f"Converged at iteration {iteration + 1}")
            break
        
        means, covariances, weights = new_means, new_covariances, new_weights
    
    return means, covariances, weights

###### Step 4: Plot Gaussian Contours ######

def plot_gaussian_contours(data, means, covariances):
    x, y = np.meshgrid(np.linspace(-10, 20, 100), np.linspace(-10, 20, 100))
    pos = np.dstack((x, y))

    plt.scatter(data[:, 0], data[:, 1], s=5)
    for mean, cov in zip(means, covariances):
        rv = multivariate_normal(mean, cov)
        plt.contour(x, y, rv.pdf(pos), levels=5, colors='red')
    
    plt.title("Gaussian Contours After EM")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()

###### Step 5: Run EM Algorithm and Plot ######

k = 3  # Number of clusters
means, covariances, weights = em_algorithm(data, k)
plot_gaussian_contours(data, means, covariances)
