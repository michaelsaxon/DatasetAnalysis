import click
import random

SNLI_PATH = "/Users/mssaxon/data/snli_1.0/"
ANLI_PATH = "/Users/mssaxon/data/anli_v1.0/"
MNLI_PATH = "/Users/mssaxon/data/multinli_1.0/"


VALID_DATASETS = {
    "S" : ("snli", None, ["sentence1", "sentence2"]),
    "A1": ("anli", 1, ["context", "hypothesis"]),
    "A2": ("anli", 2, ["context", "hypothesis"]),
    "A3": ("anli", 3, ["context", "hypothesis"]),
    "M" : ("mnli", None, ["sentence1", "sentence2"])
}


@click.command()
@click.option('--basepath', default=SNLI_PATH)
@click.option('--dataset', default="S")
@click.option('--partition', default="train")
@click.option('--outpath', default="")
@click.option('--number', default=75000)
def main(basepath, dataset, partition, outpath, number):
    ds, r, sentencemap = VALID_DATASETS[dataset]

    if ds == "anli" and basepath == SNLI_PATH:
        basepath = ANLI_PATH
    elif ds == "mnli" and basepath == SNLI_PATH:
        basepath = MNLI_PATH

    registered_path = {
        'snli_train': basepath + "snli_1.0_" + "train.jsonl",
        'snli_dev': basepath + "snli_1.0_" + "dev.jsonl",
        'snli_test': basepath + "snli_1.0_" + "test.jsonl",
        'anli_train': basepath + f"R{r}/" + "train.jsonl",
        'anli_dev': basepath + f"R{r}/" + "dev.jsonl",
        'anli_test': basepath + f"R{r}/" + "test.jsonl",
        'mnli_train': basepath + f"multinli_1.0_" + "train.jsonl",
        'mnli_dev': basepath + f"multinli_1.0_" + "dev.jsonl",
        'mnli_test': basepath + f"multinli_1.0_" + "test.jsonl",
    }

    lines = open(registered_path[f"{ds}_{partition}"]).readlines()

    random.shuffle(lines)

    lines = lines[0:number]

    with open(f"{outpath}_{dataset}_{partition}_r{int(number/1000)}k.jsonl", "w") as outfile:
        outfile.writelines(lines)


if __name__ == "__main__":
    main()