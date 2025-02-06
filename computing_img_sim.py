import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics.pairwise import cosine_similarity

def compute_metrics(img1, img2):
    
    # Compute Mean Squared Error (MSE)
    mse_value = np.mean((img1.astype("float") - img2.astype("float")) ** 2)

    # Compute Structural Similarity Index (SSIM)
    ssim_value = ssim(img1, img2)

    # Compute Cosine Similarity:
    # Flatten the grayscale images into 1D vectors
    img1_flat = img1.flatten().reshape(1, -1)
    img2_flat = img2.flatten().reshape(1, -1)
    cos_sim_value = cosine_similarity(img1_flat, img2_flat)[0][0]

    return mse_value, ssim_value, cos_sim_value

if __name__ == "__main__":
    # Load the images using OpenCV.
    # Update the paths to your image files.
    image1_path = "image1.png"
    image2_path = "image2.png"
    
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    # Check if the images were loaded successfully
    if img1 is None or img2 is None:
        raise FileNotFoundError("One or both image paths are incorrect or the images could not be loaded.")
    
    # Compute metrics between the two images
    mse_value, ssim_value, cos_sim_value = compute_metrics(img1, img2)
    
    print(f"MSE: {mse_value:.4f}")
    print(f"SSIM: {ssim_value:.4f}")
    print(f"Cosine Similarity: {cos_sim_value:.4f}")
