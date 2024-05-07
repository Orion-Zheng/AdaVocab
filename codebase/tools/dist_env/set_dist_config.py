"""
Example Usage:
srun -N 2 python set_dist_config.py --n_node 2 --gpu_per_node 2 \
                                    --main_ip x1000c1s2b0n1 \
                                    --default_config config/accelerate_config/nscc/multi_node_multi_gpu_zero3_offload.yaml \
                                    --local_save_path /tmp/dist_config.yaml
"""
import subprocess
import json
import argparse
import yaml
import os

def get_hostnames(node_count):
    """
    Execute the MPI command 'mpirun' with a specified number of nodes and capture the hostname output.

    Args:
        node_count (int): The number of nodes to use with the 'mpirun' command.

    Returns:
        list: A list of strings, each representing a line of output from the command execution,
              specifically the hostname of each node involved in the MPI task.
    """
    # Construct the command with the provided node_count
    command = ['mpirun', '-np', str(node_count), 'hostname']
    
    try:
        # Execute the command and capture the output
        result_1 = subprocess.run(command, text=True, capture_output=True, check=True)
        # Split the output by newline to get a list of lines
        hostname_list = result_1.stdout.strip().split('\n')
        
        return hostname_list
    except subprocess.CalledProcessError as e:
        # In case of an error, return the error message as a list
        raise OSError(f"Command execution failed: {e}")

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
    parser.add_argument('--n_node', type=str, help='Number of Nodes')
    parser.add_argument('--gpu_per_node', type=str, help='GPU per node')
    parser.add_argument('--main_ip', type=str, help='IP Address of the main host.')
    parser.add_argument('--default_config', type=str, help='default accelerate config')
    parser.add_argument('--local_save_path', default='my_config.yaml', type=str, help='default accelerate config')
    args = parser.parse_args()
    
    n_node = args.n_node
    gpu_per_node = args.gpu_per_node
    main_ip = args.main_ip
    default_config = args.default_config
    local_save_path = args.local_save_path

    hostname_list = get_hostnames(n_node)
    hostname_list.remove(main_ip)
    hostname_list.sort()
    
    hostname_map = {main_ip: 0}
    for idx, i in enumerate(hostname_list, 1):
        hostname_map[i] = idx
    print("Global HOST_MAP: ", hostname_map)
    
    try:
        hostname = subprocess.check_output(["hostname"], text=True).strip()
        machine_rank = hostname_map[hostname]
    except KeyError as e:
        print(f"Can't Find the Hostname in the Mapping File.")

    modifications = {
        'main_process_ip': main_ip,
        'machine_rank': int(machine_rank),
        'num_machines': int(n_node),
        'num_processes': int(n_node) * int(gpu_per_node)
    }
    print(f"Host: {hostname} \n Modifications: {modifications}")

    modify_yaml_config(default_config, local_save_path, modifications)

if __name__ == '__main__':
    main()
    