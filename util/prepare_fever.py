import os
import click
import shutil
import json
from tqdm import tqdm

@click.command()
@click.option('--path')
def main(path):
    assert os.path.isdir(path)
    shutil.copy(path + "/train_fitems.jsonl", path + "/train_labels.jsonl")
    # open the test
    lines = open(f"{path}/fever_orig/paper_dev.jsonl").readlines()
    lines += open(f"{path}/fever_orig/shared_task_dev.jsonl").readlines()
    labels = {}
    for line in lines:
        line = json.loads(line.strip())
        print(line["id"])
        labels[str(line["id"])] = line["label"]
    lines = open(path + "/dev_fitems.jsonl").readlines()
    for idx in tqdm(range(len(lines))):
        line = json.loads(lines[idx].strip())
        line["label"] = labels[str(line["cid"])]
        lines[idx] = json.dumps(line).strip() + "\n"
    with open(path + "/dev_labels.jsonl", "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    main()