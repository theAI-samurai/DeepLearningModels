import os
import configparser
from os.path import dirname, realpath
from datetime import datetime

dir_of_file = dirname(realpath(__file__))
dir_of_file = dir_of_file.replace("\\", '/')

# Reading Config.Txt file
config = configparser.RawConfigParser()
configFilePath = dir_of_file + '/conf.txt'
config.readfp(open(configFilePath))

TRAINING_DURATION_PERIOD = config.get('TRAINING_PERIOD', 'days')

# Paths
unknown_objects_folder = dir_of_file + '/ship/unknown_objects'
train_file_path = dir_of_file + '/ship/cfg/train.txt'

# Training Flag
Flag = False


def train_file_creation():
    if os.path.exists(train_file_path):
        os.remove(train_file_path)
    train_file = open(train_file_path, mode='w', encoding='utf-8')
    for file in os.listdir(unknown_objects_folder):
        if file.endswith('.jpg'):
            name = file.split('.jpg')[0]
            new_name = name + '.txt'
            if os.path.exists(unknown_objects_folder + '/' + new_name):
                jpg_files_path = unknown_objects_folder + '/'+file
                modified_date_of_file = datetime.fromtimestamp(os.stat(unknown_objects_folder + '/'+file).st_mtime)
                how_old_is_file = (datetime.now() - modified_date_of_file).days
                if how_old_is_file < int(TRAINING_DURATION_PERIOD):
                    train_file.write(jpg_files_path)
                    train_file.write('\n')
    train_file.close()


def train_script():
    train_file_creation()
    os.system("darknet_11_0.exe detector train ship/cfg/ship.data ship/cfg/yolov3_full_ship.cfg ship/backup/yolov3_full_ship_last.weights -map")


while not Flag:
    curr_time = datetime.now()
    if curr_time.hour > 0 and curr_time.hour < 5:
        Flag = True
        train_script()

