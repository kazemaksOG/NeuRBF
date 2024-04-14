#!/bin/bash

update_param() {
    local param_name=$1
    local param_value=$2
    local full_param_name=$param_name

    if [[ ! $param_name =~ ^config\. ]]; then
        full_param_name="config.$param_name"
    fi

    sed -i "s|^$full_param_name = .*|$full_param_name = $param_value|g" "$base_config"
}

restore_base_values() {
    for param in "${!base_values[@]}"; do
        base_value="${base_values[$param]}"
        sed -i "s/^config.$param = .*/config.$param = $base_value/g" "$base_config"
    done
}


base_config="./configs/img.py"

rbf_type_values=("'ivq_f'" "'ivq_d'" "'ivq_s'")
n_kernel_values=(16 64 128)
point_nn_kernel_values=(2 8 16)
ks_alpha_values=(0.5 1.25 2.0)
n_hidden_fl_values=(16 64 128)
num_layers_values=(2 3 4)
lc_act_values=("'relu'" "'sigmoid'")
act_values=("'leaky_relu'" "'tanh'")
lc_init_values=('[1e-5]' '[1e-3]' '[1e-2]')
lcb_init_values=('[1e-5]' '[1e-3]' '[1e-2]')
w_init_values=("['xavier_uniform', 'xavier_uniform', 'xavier_uniform']" "['kaiming_uniform','kaiming_uniform','kaiming_uniform']")
b_init-values=("['zeros','zeros','zeros']" "['ones','ones','ones']")
a_init_values=('[6,30,30,30]' '[12,30,30,30]' '[18,30,30,30]' '[9,20,20,40]' '[15,50,50,50]' '[9,40,20,20]' '[9,20,40,20]')
val_freq_values=(0.5 1.25 2.0)
lr_values=('1e-3' '1e-2' '1e-1')

declare -A base_values
base_values=(
    [rbf_type]="'ivq_a'"
    [n_kernel]="'auto'"
    [point_nn_kernel]="4"
    [ks_alpha]="1"
    [n_hidden_fl]="32"
    [num_layers]="3"
    [lc_act]="'none'"
    [act]="'relu'"
    [lc_init]="[1e-4]"
    [lcb_init]="[1e-4]"
    [w_init]="[None, None, None]"
    [b_init]="[None, None, None]"
    [a_init]="[9, 30, 30, 30]"
    [val_freq]="1.0"
    [lr]="0.005"
)

params=(rbf_type n_kernel point_nn_kernel ks_alpha n_hidden_fl num_layers lc_act act lc_init lcb_init w_init b_init a_init max_steps val_freq lr)

for param in "${params[@]}"; do
    values=("${param}_values[@]")
    for value in "${!values}"; do
        update_param "$param" "$value"

        echo "Running model with $param = $value"

        python main.py --config "$base_config" --path ./data/img/pluto1000.png --alias "pluto_${param}_$value"
    done
    restore_base_values
done
