import numpy as np
import sys
import matplotlib.pyplot as plt
import os
plt.rcParams.update({'font.size': 18})

def plot_psnr(psnr_1, psnr_2, psnr_3):
    try:
        # Load array from file
        psnr_1_array = np.load(psnr_1)
        psnr_2_array = np.load(psnr_2)
        psnr_3_array = np.load(psnr_3)
        
        # Extract x and y data
        x_1 = psnr_1_array[:, 0]  # Assuming the first column contains the step count
        y_1 = psnr_1_array[:, 1]  # Assuming the second column contains the PSNR values
        
        x_2 = psnr_2_array[:, 0]  # Assuming the first column contains the step count
        y_2 = psnr_2_array[:, 1]  # Assuming the second column contains the PSNR values

        x_3 = psnr_3_array[:, 0]  # Assuming the first column contains the step count
        y_3 = psnr_3_array[:, 1]  # Assuming the second column contains the PSNR values

        plt.figure(figsize=(8, 6))
        plt.plot(x_1, y_1, linestyle='-', label="lr=0.005")
        plt.plot(x_3, y_3, linestyle='-', label="lr=0.001")
        plt.plot(x_2, y_2, linestyle='-', label="lr=0.1")
        plt.title('PSNR vs Step Count')
        plt.xlabel('Step Count')
        plt.ylabel('PSNR')
        plt.grid(True)
        plt.legend()
        plt.show()
        
    except FileNotFoundError:
        print("File not found. Please provide a valid file path.")
    except Exception as e:
        print("An error occurred:", e)

def plot_multiple(directory):
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Check if the file is an image (you can add more image extensions if needed)
        if filename.endswith("psnr.txt"):
            # Get the full path of the image
            psnr = os.path.join(directory, filename)
            print(psnr) 
            # plot_psnr("./results/results/colorful_3.5k-rbf-v0_psnr.txt", psnr)

if __name__ == "__main__":
    if len(sys.argv) < 2: 
        print("Usage: psnr_graph.py [psnr.txt]")
       
    elif len(sys.argv) == 2:
        file_path = sys.argv[1]
        plot_multiple(file_path)
    else:
        original = sys.argv[1]
        compare = sys.argv[2]
        compare_2 = sys.argv[3]
        plot_psnr(original, compare, compare_2)

