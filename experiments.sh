#!/bin/bash

# Define the scripts to run
scripts=("ablations.sh" "ablations_comb.sh")

# Loop over each script
for script in "${scripts[@]}"; do
    # Check if the script exists
    if [[ -f "$script" ]]; then
        # Check if the script is executable
        if [[ -x "$script" ]]; then
            echo "Running $script"
            # Run the script
            bash "$script"
        else
            echo "$script is not executable"
            echo "Trying to add execute permissions to $script"
            chmod +x "$script"
            if [[ -x "$script" ]]; then
                echo "Running $script"
                # Run the script
                bash "$script"
            else
                echo "Failed to add execute permissions to $script"
            fi
        fi
    else
        echo "$script does not exist"
    fi
done