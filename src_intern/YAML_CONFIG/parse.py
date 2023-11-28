'''Modules'''
from typing import List, Tuple
import yaml

def yaml_to_env(config_file: str) -> str:
    '''
    Convert YAML string into a list of ENV variables
    '''
    yaml_dict: dict = yaml.safe_load(config_file)

    def recurse_yaml(yaml_dict: dict, key_path: List[str]) -> str:
        '''
        Recursion cycle
        (key path is a chain of keys leading to an eventual value)
        '''
        yaml_str = ''
        # going over all key-value pairs
        for key, value in yaml_dict.items():
            key_path.append(key)
            # if the value is nested dict, go over it recursively
            if isinstance(value, dict):
                yaml_str += f'{recurse_yaml(value, key_path[:])}'
            # otherwise, add a new var
            else:
                yaml_str += f'{".".join(key_path)}={value}\n'

            key_path = key_path[:-1] # strip off the last key

        return yaml_str

    env_list = recurse_yaml(yaml_dict, key_path=[])
    return env_list

def env_to_yaml(env_list: str) -> str:
    '''
    Convert a list of ENV variables into YAML non-quoted string
    '''
    def line_to_kv(line: str) -> Tuple[List, str]:
        '''
        Convert a line with var into (key_path: list, value: str)
        '''
        keys, val = line.split('=')
        key_path = keys.split('.')
        return key_path, val

    yaml_str: str = ''

    # getting the list of env vars (key paths and values)
    env_vars = env_list.split('\n')
    env_vars = list(filter(lambda var: var != '', env_vars))
    env_vars = list(map(line_to_kv, env_vars))

    for key_path, value in env_vars:
        # format into YAML (accounting for indentation and key decomposition)
        key_path_str = ''.join([str('  ' * (i) + key_path[i] + ':\n')
                                for i in range(len(key_path)-1)])
        yaml_str += key_path_str + '  ' * (len(key_path)-1) + key_path[-1] + f': {value}\n'

    # remove duplicate lines
    # (priorly remove the last line which is empty)
    yaml_list = yaml_str.split('\n')[:-1]
    unique_list = []
    _ = [unique_list.append(var) for var in yaml_list if var not in unique_list]
    yaml_str = '\n'.join(unique_list)

    return yaml_str
