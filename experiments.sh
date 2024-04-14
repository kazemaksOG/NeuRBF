#!/bin/bash

scripts=("ablations.sh" "ablations_comb.sh")

for script in "${scripts[@]}"; do
    if [[ -f "$script" ]]; then
        if [[ -x "$script" ]]; then
            echo "Running $script"
            bash "$script"
        else
            echo "$script is not executable"
            echo "Trying to add execute permissions to $script"
            chmod +x "$script"
            if [[ -x "$script" ]]; then
                echo "Running $script"
                bash "$script"
            else
                echo "Failed to add execute permissions to $script"
            fi
        fi
    else
        echo "$script does not exist"
    fi
done