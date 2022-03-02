import os
import click
import shutil

@click.command()
@click.option('--ocdir')
def main(ocdir):
    assert os.path.isdir(ocdir)
    # mv test.json test_unlabeled.json
    shutil.move(ocdir + "/test.json", ocdir + "test_unlabeled.json")
    # mv dev.json test.json
    shutil.move(ocdir + "/dev.json", ocdir + "test.json")
    # cp train.3k.json dev.json
    shutil.copy(ocdir + "/train.3k.json", ocdir + "dev.json")
    # remove overlapping 3k sentences from 50k
    train_lines = open(ocdir + "/train.50k.json", "r").readlines()
    dev_lines = open(ocdir + "/dev.json", "r").readlines()
    head = train_lines[0]
    train_lines = train_lines[1:]
    dev_lines = dev_lines[1:]
    for line in dev_lines:
        if line in train_lines:
            train_lines.remove(line)
    with open(ocdir + "/train.json", "w") as f:
        f.writelines([head] + train_lines)


if __name__ == "__main__":
    main()