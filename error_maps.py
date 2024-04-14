from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
import imageio.v2 as imageio
import io
import os

def generate_error_map(image1_path, image2_path):
    image1 = Image.open(image1_path).convert('RGB')
    image2 = Image.open(image2_path).convert('RGB')

    array1 = np.array(image1)
    array2 = np.array(image2)

    diff = np.abs(array1 - array2)
    mse = np.mean(np.square(array1 - array2))

    error_map = np.mean(diff, axis=2)

    return error_map, mse


if __name__ == "__main__":
    if len(sys.argv) < 2: 
        print("Usage: error_map.py [original_file] [input_folder] [output_folder]")
    else:
        original = sys.argv[1]
        input_folder = sys.argv[2]
        output_folder = sys.argv[3]

        files = os.listdir(input_folder)
        files = [os.path.join(input_folder, file) for file in files if file.endswith('.png')]
        files = sorted(files)

        BIG_GRID = True

        if not BIG_GRID: 
            for i, file in enumerate(files):
                error_map, mse = generate_error_map(original, file)
                print(f"Mean Squared Error (MSE) for {original} and {file}: {mse}")

                name = file.split('/')[-1]
                name = name.split('.')[0]

                plt.imsave(f'{output_folder}/{name}_em.png', error_map, cmap='jet')
        else:
            error_maps = []
            for i, file in enumerate(files):
                error_map, mse = generate_error_map(original, file)
                print(f"Mean Squared Error (MSE) for {original} and {file}: {mse}")
                name = file.split('/')[-1]
                name = name.split('.')[0]
                error_maps.append((error_map, name))
            
            n = len(error_maps)
            cols = int(np.sqrt(n))
            rows = int(np.ceil(n / cols))

            fig, axs = plt.subplots(rows, cols, figsize=(cols * 10, rows * 10))

            plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.5, wspace=0.5)
            
            for i, (error_map, name) in enumerate(error_maps):
                ax = axs[i // cols, i % cols]
                ax.imshow(error_map, cmap='jet')
                ax.set_title(name)
                ax.axis('off')

            for j in range(i+1, rows*cols):
                ax = axs[j // cols, j % cols]
                ax.axis('off')
            plt.savefig(f'{output_folder}/error_map_grid.png')


            INTERP = True
            if INTERP:
                fig, axs = plt.subplots(rows, cols, figsize=(cols * 10, rows * 10))
                plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.5, wspace=0.5)
            
                for i, (error_map, name) in enumerate(error_maps):
                    ax = axs[i // cols, i % cols]
                    ax.imshow(error_map, cmap='jet', interpolation='gaussian')
                    ax.set_title(name)
                    ax.axis('off')

                for j in range(i+1, rows*cols):
                    ax = axs[j // cols, j % cols]
                    ax.axis('off')

                plt.savefig(f'{output_folder}/error_map_grid_interp.png')

        fig, ax = plt.subplots(figsize=(6, 1))
        fig.subplots_adjust(bottom=0.5)

        cmap = plt.get_cmap('jet')

        cb1 = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=ax, orientation='horizontal')

        cb1.set_label('Error')
        plt.savefig(f'{output_folder}/colorbar.png')

            
    