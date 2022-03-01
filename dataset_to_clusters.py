import click

from training.train_classifier import *
from training.manage_settings import get_write_settings, lazymkdir

def collect_embeddings(dataset):
    



@click.command()
@click.option('--n_gpus', default=1, help='number of gpus')
@click.option('--dataset', help="S, M, A1, A2, A3, OC, SICK, etc")
@click.option('--model_id', help="path to trained model or name of hf pretrained")
@click.option('--batch_size', default=48)
@click.option('--biased', is_flag=True)
@click.option('--extreme_bias', is_flag=True)
@click.option('--s2only', is_flag=True)
def main(n_gpus, dataset, biased, model_id, batch_size, extreme_bias, s2only):
    model, tokenizer = choose_load_model_tokenizer(model_id, dataset)
    print("Init litmodel...")
    ltmodel = RobertaClassifier(model)
    print("Init dataset...")
    if extreme_bias:
        factor = 0
    else:
        factor = 1
    nli_data = plNLIDataModule(tokenizer, dir_settings["dataset_dir"], dataset, batch_size, biased, factor, s2only)
    dir_settings = get_write_settings(["data_save_dir", "dataset_dir"])

    embeddings = collect_embeddings(nli_data, ltmodel)


if __name__ == "__main__":
    main()