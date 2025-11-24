import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

# Step 1: Image processing
image = cv2.imread('buzz.jpeg', cv2.IMREAD_GRAYSCALE)
flattened_image = image.flatten()

plt.imshow(image, cmap='gray')
plt.title('Grayscale Image')
plt.show()

# Step 2: Computing the Covariance matrix
mean = np.mean(flattened_image)

normalized_image = flattened_image - mean
normalized_image_reshaped = normalized_image.reshape(image.shape)

cov_matrix = np.cov(normalized_image_reshaped, rowvar=False)

print("Covariance Matrix:\n", cov_matrix)

# Step 3: Find the Eigenvalues and Eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
print("Eigenvalues:\n", eigenvalues)
print("\nEigenvectors:\n", eigenvectors)

# Step 4: Select the PCA with the desired variance
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

# Select top k principal components
k = 50  # You can change this value
top_k_eigenvectors = sorted_eigenvectors[:, :k]

print("Top k Eigenvectors:\n", top_k_eigenvectors)

# Step 5: Reconstruct the image
projected_image = np.dot(normalized_image_reshaped, top_k_eigenvectors)
reconstructed_image = np.dot(projected_image, top_k_eigenvectors.T) + mean
reconstructed_image = reconstructed_image.reshape(image.shape)

plt.imshow(reconstructed_image, cmap='gray')
plt.title('Reconstructed Image using PCA')
plt.show()

# Alternative: Using scikit-learn for PCA

# Flatten image for PCA (each row is a sample, each column is a feature)
image_flat = image.astype(float)

# Fit PCA
pca = PCA(n_components=k)
pca_transformed = pca.fit_transform(image_flat)
reconstructed = pca.inverse_transform(pca_transformed)

plt.imshow(reconstructed, cmap='gray')
plt.title('Reconstructed Image using scikit-learn PCA')
plt.show()