from train_classifier import *
from dataset_to_clusters import *
from manage_settings import get_write_settings, read_models_csv, lazymkdir

def collect_posteriors(nli_dataset, ltmodel):
    print("Collecting decisions...")
    for batch in tqdm(nli_dataset.test_dataloader()):
        cuda_dict(batch)
        batch_embs = ltmodel(batch)
        yield batch_embs, batch["labels"]

def 

def main():
    embs, labs = collect_choices()