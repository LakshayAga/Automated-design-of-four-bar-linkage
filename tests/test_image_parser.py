import os
import sys
import cv2
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.image_parser import extract_path_from_image

def create_synthetic_image(path):
    # Create white canvas
    img = np.ones((500, 500), dtype=np.uint8) * 255
    # Draw a black circle to simulate a drawn curve
    cv2.circle(img, (250, 250), 100, 0, 5) # Center, radius, color (black), thickness
    cv2.imwrite(path, img)

def test():
    test_img_path = "synthetic_test.png"
    create_synthetic_image(test_img_path)
    
    try:
        tensor = extract_path_from_image(test_img_path)
        print("Success! Parsed tensor successfully.")
        print(f"Tensor Shape: {tensor.shape}")
        print(f"Tensor Type: {tensor.dtype}")
        print(tensor)
    finally:
        if os.path.exists(test_img_path):
            os.remove(test_img_path)

if __name__ == "__main__":
    test()
