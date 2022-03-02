from pathlib import PurePath
import os
from typing import List
import glob

def lazymkdir(file):
    head = os.path.split(file)[0]
    if not os.path.isdir(head):
        os.mkdir(head)
    return file

def get_write_setting(requested_setting, settings_dir):
    lazymkdir(settings_dir)
    setting_path = PurePath(settings_dir + "/" + requested_setting)
    if not os.path.exists(setting_path):
        setting_val = input(f"Please provide requested setting '{requested_setting}':")
        with open(setting_path, "w") as f:
            f.write(setting_val)
    else:
        with open(setting_path, "r") as f:
            setting_val = f.read()
    return setting_val

def get_write_settings(requested_settings : List[str], settings_dir = ".settings/"):
    settings_vals = {}
    for setting in requested_settings:
        settings_vals[setting] = get_write_setting(setting, settings_dir)
    return settings_vals

def read_models_csv(dataset, csv_path = "finetuned_models.csv"):
    if os.path.exists(csv_path):
        lines = open(csv_path, "r").readlines()
        for line in lines:
            line = line.strip().split(",")
            if dataset == line[0]:
                return line[1], line[2]
    # csv doesn't exist or dataset not in it.
    model_name = input(f"Please provide huggingface model_id (default roberta-large):")
    if model_name == "":
        model_name = "roberta-large"
    pretrained_path = input(f"Please provide pretrained model path (default=no pretrained):")
    if pretrained_path[-4:] != "ckpt":
        potential_checkpoints = glob.glob(pretrained_path + "/checkpoints/*")
        pretrained_path = potential_checkpoints[-1]
    return model_name, pretrained_path