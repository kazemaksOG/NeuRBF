#!/bin/bash

# Define the values for ablatable parameters related to network architecture
rbf_types=('gaussian' 'inverse_multiquadric' 'thin_plate')
n_kernel_values=(16 64 128)
point_nn_kernel_values=(2 8 16)
num_layers_values=(2 3 4)

# Loop through combinations of selected network architecture parameters
for rbf_type in "${rbf_types[@]}"; do
    for n_kernel in "${n_kernel_values[@]}"; do
        for point_nn_kernel in "${point_nn_kernel_values[@]}"; do
            for num_layers in "${num_layers_values[@]}"; do
                # Update the config file with the current parameter values
                sed -i "s/config.rbf_type = .*/config.rbf_type = '$rbf_type'/" configs/img.py
                sed -i "s/config.n_kernel = .*/config.n_kernel = $n_kernel/" configs/img.py
                sed -i "s/config.point_nn_kernel = .*/config.point_nn_kernel = $point_nn_kernel/" configs/img.py
                sed -i "s/config.num_layers = .*/config.num_layers = $num_layers/" configs/img.py
                
                # Run the model
                python main.py --config configs/img.py --path ./data/img/pluto1000.png --alias pluto_"$rbf_type"_"$n_kernel"_"$point_nn_kernel"_"$num_layers"
                
                # Revert the config file back to the base state
                cp configs/base_img.py configs/img.py
            done
        done
    done
done
