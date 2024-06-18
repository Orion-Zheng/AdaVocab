"""
Example Usage:
python codebase/tools/dist_env/set_dist_config_new.py \
                        --config_template ${ACCELERATE_CONFIG} \
                        --global_rank ${GLOBAL_RANK} \
                        --main_proc_ip ${MASTER_ADDR} \
                        --main_proc_port ${MASTER_PORT} \
                        --n_node ${N_NODE} \
                        --world_size ${WORLD_SIZE} \
                        --local_save_path ${SAVE_CONFIG_DIR} + "/" + $(hostname) + ".yaml"
"""
import subprocess
import json
import argparse
import yaml
import os
import socket


def modify_yaml_config(input_file_path, output_file_path, modifications):
    # Make sure the output path exist
    if os.path.dirname(output_file_path):
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    # Read the .yaml config
    with open(input_file_path, 'r') as file:
        config = yaml.safe_load(file)

    # Apply modifications
    for key, value in modifications.items():
        config[key] = value

    # Save the modified .yaml config to output_file_path
    with open(output_file_path, 'w') as file:
        yaml.safe_dump(config, file, sort_keys=False)

    print(f'Modified Config has been saved to: {output_file_path}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_template', type=str, help='default accelerate config')
    parser.add_argument('--global_rank', type=str, help='Global GPU Rank')
    parser.add_argument('--main_proc_ip', default=None, type=str, help='IP Address of the main host.')
    parser.add_argument('--main_proc_port', default=None, type=str, help='communication port of the main process')
    parser.add_argument('--n_node', type=str, help='Number of nodes')
    parser.add_argument('--world_size', type=str, help='World GPU Size')
    parser.add_argument('--local_save_path', default='my_config.yaml', type=str, help='default accelerate config')
    args = parser.parse_args()
    
    main_ip = args.main_proc_ip
    main_port = int(args.main_proc_port) if args.main_proc_port is not None else None

    global_rank = int(args.global_rank)
    n_node = int(args.n_node)
    world_size = int(args.world_size)

    config_template = args.config_template
    local_save_path = args.local_save_path

    modifications = {
        'machine_rank': global_rank,  
        'num_machines': n_node,  
        'num_processes': world_size
    }
    if main_ip is not None:
        modifications['main_process_ip'] = main_ip
    if main_port is not None:
        modifications['main_process_port'] = main_port
    hostname = socket.gethostname()
    print(f"Host: {hostname} \n Modifications: {modifications}")

    modify_yaml_config(config_template, local_save_path, modifications)

if __name__ == '__main__':
    main()
    