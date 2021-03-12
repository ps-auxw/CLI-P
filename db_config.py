import os
import configparser

vectors_map_size = 1024*1024*1024*20
config_map_size = 1024*1024*1024

config = configparser.ConfigParser()
config.read('cli-p.conf')
if 'DEFAULT' in config:
    if 'vectors_map_size' in config['DEFAULT']:
        vectors_map_size = int(config['DEFAULT']['vectors_map_size'])
    if 'config_map_size' in config['DEFAULT']:
        config_map_size = int(config['DEFAULT']['config_map_size'])
