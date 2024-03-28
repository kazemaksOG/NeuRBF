from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
import imageio.v2 as imageio
import io

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


def save_error_map(error_map):
    buf = io.BytesIO()

    plt.imshow(error_map, cmap='jet', interpolation='gaussian')
    plt.colorbar()
    plt.title('Error Map')
    plt.savefig(buf, format='png')
    plt.close()

    buf.seek(0)
    img = Image.open(buf)
    return np.array(img)


if __name__ == "__main__":
    if len(sys.argv) < 2: 
        print("Usage: error_map.py [original_file] [folder]")
    else:
        original = sys.argv[1]
        folder = sys.argv[2]

        import os
        files = os.listdir(folder)
        files = [os.path.join(folder, file) for file in files if file.startswith('step_')]
        files = sorted(files)
        

        # Create MP4 from error map images
        with imageio.get_writer('error_maps.mp4', mode='I') as writer:
            for i, file in enumerate(files):
                error_map, mse = generate_error_map(original, file)
                print(f"Mean Squared Error (MSE) for {original} and {file}: {mse}")
                
                img_array = save_error_map(error_map)

                # Save the first image to a file
                if i == 0:
                    img = Image.fromarray(img_array)
                    img.save('first_image.png')

                # Append image to video
                writer.append_data(img_array)

