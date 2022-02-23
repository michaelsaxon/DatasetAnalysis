from pathlib import PurePath
import os
from typing import List

def lazymkdir(file):
    head = os.path.split(file)[0]
    if not os.path.isdir(head):
        os.mkdir(head)
    return file

def get_write_setting(requested_setting, settings_dir):
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