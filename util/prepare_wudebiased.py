'''
python prepare_wudebiased.py --fname snli_z-aug.jsonl
'''
import os
import click
import shutil
import random
import json

@click.command()
@click.option('--fname')
@click.option('--val_split', default = .2)
@click.option('--basedir', default = "/local/home/saxon/data/debiased_wu/")
def main(fname, val_split, basedir):
    fname = basedir + fname
    assert os.path.exists(fname)
    # mv test.json test_unlabeled.json
    #shutil.move(ocdir + "/test.json", ocdir + "/test_unlabeled.json")
    # mv dev.json test.json
    # cp train.3k.json dev.json
    #shutil.copy(ocdir + "/train.3k.json", ocdir + "/dev.json")
    # remove overlapping 3k sentences from 50k
    lines = open(fname, "r").readlines()
    # remove .jsonl 
    out_fname_base = fname[:-6]
    train_lines = []
    val_lines = []
    val_hyps = {}
    val_prems = {}
    random.shuffle(lines)
    target_val_length = len(lines) * val_split
    for i, line in enumerate(lines):
        _line = json.loads(line)
        if len(val_lines) < target_val_length:
            # add to val lines
            val_lines.append(line)
            val_hyps[_line["hypothesis"]] = True
            val_prems[_line["premise"]] = True
        else:
            # check if we can add this to train (we can't if there's a collision)
            if val_hyps.get(_line["hypothesis"], False) or val_prems.get(_line["premise"], False):
                val_lines.append(line)
            else:
                train_lines.append(line)
    with open(out_fname_base+"_train.jsonl", "w") as f:
        f.writelines(train_lines)
    with open(out_fname_base + "_dev.jsonl", "w") as f:
        f.writelines(val_lines)


if __name__ == "__main__":
    main()