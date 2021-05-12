import configparser
from os.path import dirname, realpath

dir_of_file = dirname(realpath(__file__))
dir_of_file = dir_of_file.replace("\\", '/')

# Reading Config.Txt file
config = configparser.RawConfigParser()
configFilePath = dir_of_file + '/conf.txt'
config.readfp(open(configFilePath))

all_camera_data = {}            # saves ID : url , of cameras in a dictionary after reading from config file.

# Resolution
WIDTH = config.get('RESOLUTION', 'WIDTH')
HEIGHT = config.get('RESOLUTION', 'HEIGHT')

# Extracting fields and values for cameras
try:
    camera_path_1 = config.get('camera_1', 'feeder_path')
    camera_id_1 = config.get('camera_1', 'param_id')
    all_camera_data.update({camera_id_1: camera_path_1})
    del (camera_id_1, camera_path_1)
except:
    print('Please check Camera 1 in conf.txt file !!')
try:
    camera_path_2 = config.get('camera_2', 'feeder_path')
    camera_id_2 = config.get('camera_2', 'param_id')
    all_camera_data.update({camera_id_2: camera_path_2})
    del (camera_id_2, camera_path_2)
except:
    print('Please check Camera 2 in conf.txt file !!')
try:
    camera_path_3 = config.get('camera_3', 'feeder_path')
    camera_id_3 = config.get('camera_3', 'param_id')
    all_camera_data.update({camera_id_3: camera_path_3})
    del (camera_id_3, camera_path_3)
except:
    print('Please check Camera 3 in conf.txt file !!')
try:
    camera_path_4 = config.get('camera_4', 'feeder_path')
    camera_id_4 = config.get('camera_4', 'param_id')
    all_camera_data.update({camera_id_4: camera_path_4})
    del (camera_id_4, camera_path_4)
except:
    print('Please check Camera 4 in conf.txt file !!')


# Extracting fields for Requests
notification_url = config.get('GET_POST_URL', 'notification')
markup_base_url = config.get('GET_POST_URL', 'markup')
ref_base_url = config.get('GET_POST_URL', 'ref')
camera_status_url = config.get('GET_POST_URL', 'cam_active_status')

del(config, configFilePath)





