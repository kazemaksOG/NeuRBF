from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys

def generate_error_map(image1_path, image2_path):
    # Read the images
    image1 = Image.open(image1_path).convert('RGB')
    image2 = Image.open(image2_path).convert('RGB')

    # Convert images to NumPy arrays
    array1 = np.array(image1)
    array2 = np.array(image2)

    # Calculate pixel-wise difference
    diff = np.abs(array1 - array2)

    # Calculate mean squared error (MSE)
    mse = np.mean(np.square(array1 - array2))

    # take mean error between color channels
    error_map = np.mean(diff, axis=2)

    return error_map, mse

def display_error_map(error_map):
    plt.imshow(error_map, cmap='jet', interpolation='gaussian')
    plt.colorbar()
    plt.title('Error Map')
    plt.show()




if __name__ == "__main__":
    if len(sys.argv) < 2: 
        print("Usage: error_map.py [ORIGINAL.png] [GENERATED.png]")
       
    else:
        original = sys.argv[1]
        generated = sys.argv[2]

        error_map, mse = generate_error_map(original, generated)
        print(f"Mean Squared Error (MSE): {mse}")

        display_error_map(error_map)
