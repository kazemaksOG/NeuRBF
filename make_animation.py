import sys
import os
import imageio.v2 as imageio

if __name__ == "__main__":
    if len(sys.argv) < 2: 
        print("Usage: make_animation.py [folder]")
    else:
        folder = sys.argv[1]

        import os
        images = sorted([os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.png')])
        
        writer = imageio.get_writer('output.mp4')

        for image in images:
            writer.append_data(imageio.imread(image))

        writer.close()