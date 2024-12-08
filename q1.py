import numpy as np
import matplotlib.pyplot as plt


# Debug dosyasını başlatma
debug_file = open("debug_log_q1.txt", "w")

# Define the parameters for 3 Gaussian clusters
means = [[0, 0], [5, 5], [10, 0]]  # Cluster means
covariances = [np.eye(2) * 2, np.eye(2) * 1.5, np.eye(2) * 3]  # Isotropic covariance matrices
samples_per_cluster = [300, 400, 250]  # Number of samples per cluster

# Debug: Cluster parametrelerini yaz
debug_file.write("Cluster Parameters:\n")
for idx, (mean, cov, n_samples) in enumerate(zip(means, covariances, samples_per_cluster)):
    debug_file.write(f"Cluster {idx+1}:\n")
    debug_file.write(f"  Mean: {mean}\n")
    debug_file.write(f"  Covariance:\n{cov}\n")
    debug_file.write(f"  Number of Samples: {n_samples}\n")
debug_file.write("\n")

# Generate data for each cluster
data = []
for mean, cov, n_samples in zip(means, covariances, samples_per_cluster):
    cluster_data = np.random.multivariate_normal(mean, cov, n_samples)
    data.append(cluster_data)

data = np.vstack(data)

# Debug: Oluşturulan verileri yaz
debug_file.write("Generated Data:\n")
np.savetxt(debug_file, data, fmt="%.5f", header="X1,X2", delimiter=",")
debug_file.write("\n")


def gaussian_pdf(x, mean, cov):
    """Compute the Gaussian PDF."""
    dim = len(mean)  # Veri boyutu (örneğin 2D için 2)
    diff = x - mean  # Noktanın Gaussian merkezi ile farkı
    return (1 / (np.sqrt((2 * np.pi)**dim * np.linalg.det(cov)))) * \
           np.exp(-0.5 * np.dot(diff.T, np.dot(np.linalg.inv(cov), diff)))

# Eğer yeni merkezlerin eski merkezlerden farkı toleransın altında ise algoritma durur.
def em_algorithm(data, k, max_iter=100, tol=1e-4):
    """EM Algorithm for Gaussian Mixture Models."""
    n_samples, n_features = data.shape  # Veri boyutu

    # 1. Parametrelerin Başlatılması
    np.random.seed(42) # deneysel tekrarlanabilirlik için sabitleme
    means = data[np.random.choice(n_samples, k, replace=False)]  # Gaussian merkezleri (rastgele başlatma)
    covariances = [np.eye(n_features) for _ in range(k)]  # İzotropik kovaryans matrisleri
    weights = np.ones(k) / k  # Başlangıçta eşit ağırlıklar

 # Debug: Başlatma parametreleri
    debug_file.write("Initial Parameters:\n")
    debug_file.write(f"Initial Means:\n{means}\n")
    debug_file.write(f"Initial Covariances:\n")
    for cov in covariances:
        debug_file.write(f"{cov}\n")
    debug_file.write(f"Initial Weights: {weights}\n\n")

    for iteration in range(max_iter):
        # 2. E-Adımı (Possibilities Hesaplanması)
        probabilities = np.zeros((n_samples, k))
        for i in range(k):
            for j in range(n_samples):
                probabilities[j, i] = weights[i] * gaussian_pdf(data[j], means[i], covariances[i])
        probabilities /= probabilities.sum(axis=1, keepdims=True) # keepdims: Matris boyutunun korunmasını sağlar.

        # Debug: Sorumluluklar
        debug_file.write(f"Iteration {iteration+1} - Probabilities:\n")
        np.savetxt(debug_file, probabilities, fmt="%.5f", header="Cluster1,Cluster2,Cluster3", delimiter=",")
        debug_file.write("\n")

        # 3. M-Adımı (Parametre Güncellemesi)
        new_means = np.zeros_like(means) # zeros_like: Başlangıç için sıfır matris oluşturur.
        new_covariances = [np.zeros_like(cov) for cov in covariances]
        new_weights = np.zeros(k)

        for i in range(k): # 
            prob_sum = probabilities[:, i].sum()  # Küme i'ye ait sorumlulukların toplamı
            new_weights[i] = prob_sum / n_samples  # Yeni ağırlıklar
            new_means[i] = (probabilities[:, i][:, None] * data).sum(axis=0) / prob_sum  # Yeni ortalamalar
            diff = data - new_means[i]
            diff = data - new_means[i]  # Veri ile yeni küme merkezlerinin farkı
            cov_matrix = np.zeros_like(covariances[i])  # Sıfır matris
            for j in range(n_samples):
                cov_matrix += probabilities[j, i] * np.outer(diff[j], diff[j])  # Dış çarpım
            new_covariances[i] = cov_matrix / prob_sum  # Kovaryansı normalize et

        # Debug: Güncellenen parametreler
        debug_file.write(f"Iteration {iteration+1} - Updated Parameters:\n")
        debug_file.write(f"Updated Means:\n{new_means}\n")
        debug_file.write(f"Updated Covariances:\n")
        for cov in new_covariances:
            debug_file.write(f"{cov}\n")
        debug_file.write(f"Updated Weights: {new_weights}\n\n")

        # 4. Yakınsama Kontrolü
        if np.linalg.norm(new_means - means) < tol:
            break

        means = new_means
        covariances = new_covariances
        weights = new_weights

    return means, covariances, weights

# 5. EM Algoritmasını çalıştır
k = 3
means, covariances, weights = em_algorithm(data, k)

# 6. Gaussian konturlarını çiz
x, y = np.meshgrid(np.linspace(-5, 15, 100), np.linspace(-5, 15, 100))#
pos = np.dstack((x, y))

plt.scatter(data[:, 0], data[:, 1], s=10)
for mean, cov in zip(means, covariances):
    pdf = np.array([gaussian_pdf(np.array([x_i, y_i]), mean, cov) for x_i, y_i in zip(x.flatten(), y.flatten())])
    plt.contour(x, y, pdf.reshape(x.shape), levels=5, colors='red') # Contours: Yoğunluğu göstermek için kırmızı çizgiler.
plt.title("Gaussian Contours After EM")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

# Debug dosyasını kapat
debug_file.close()
print("Debug log saved to 'debug_log.txt'")
