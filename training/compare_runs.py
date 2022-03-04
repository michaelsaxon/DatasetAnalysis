"""
CUDA_VISIBLE_DEVICES=0 python compare_runs.py --dataset A3
"""

from train_classifier import *
from dataset_to_clusters import *
from manage_settings import get_write_settings, read_models_csv, lazymkdir

def collect_posteriors(nli_dataset, ltmodel):
    print("Collecting decisions...")
    for batch in tqdm(nli_dataset.test_dataloader()):
        cuda_dict(batch)
        batch_posts = ltmodel(batch)
        yield batch_posts, batch["labels"]

def get_numpy_preds(nli_data, ltmodel):
    decisions_list = []
    labels_list = []
    for batch_posts, batch_labs in collect_posteriors(nli_data, ltmodel):
        batch_decisions = torch.max(batch_posts, -1).indices
        batch_decisions = batch_decisions.cpu().detach().numpy()
        batch_labs = batch_labs.cpu().detach().numpy()
        decisions_list.append(batch_decisions)
        labels_list.append(batch_labs)
    decisions_list = np.concatenate(decisions_list)
    labels_list = np.concatenate(labels_list)
    return decisions_list, labels_list


@click.command()
@click.option('--n_gpus', default=1, help='number of gpus')
@click.option('--dataset', help="S, M, A1, A2, A3, OC, SICK, etc")
@click.option('--batch_size', default=48)
def main(n_gpus, dataset, batch_size):
    # do 2 diff pretrained paths
    model_id, pretrained_path_1 = read_models_csv(dataset)
    _, pretrained_path_2 = read_models_csv(dataset, True)

    print("hmmm")
    model, tokenizer = choose_load_model_tokenizer(model_id, dataset)
    print("hmmmmm.......")
    ltmodel = RobertaClassifier(model, learning_rate=0)
    print("Init litmodel...")
    ckpt_1 = torch.load(pretrained_path_1)
    ckpt_2 = torch.load(pretrained_path_2)
    ltmodel.load_state_dict(ckpt_1["state_dict"])
    print("Init dataset...")
    if extreme_bias:
        factor = 0
    else:
        factor = 1

    model.cuda()
    ltmodel.cuda()

    dir_settings = get_write_settings(["data_save_dir", "dataset_dir", "intermed_comp_dir_base"])
    
    intermed_comp_dir = setup_intermed_comp_dir(dir_settings["intermed_comp_dir_base"], dataset,
        n_clusters, (biased, extreme_bias, s2only))

    nli_data_1 = plNLIDataModule(tokenizer, dir_settings["dataset_dir"], dataset, False, False, 1, False)
    nli_data_1.prepare_data(test_only = True)

    nli_data_2 = plNLIDataModule(tokenizer, dir_settings["dataset_dir"], dataset, False, False, 1, True)
    nli_data_2.prepare_data(test_only = True)


    dec_1, labs_1 = get_numpy_preds(nli_data_1, ltmodel)

    ltmodel.load_state_dict(ckpt_2["state_dict"])
    model.cuda()
    ltmodel.cuda()

    dec_2, labs_2 = get_numpy_preds(nli_data_2, ltmodel)

    agreement = np.equal(dec_1, dec_2)
    correct_1 = np.equal(dec_1, labs_1)
    correct_2 = np.equal(dec_2, labs_2)
    correct_agreement = agreement * correct_1 * correct_2

    print(agreement.sum() / agreement.shape[0])
    print(correct_agreement.sum() / correct_agreement.shape[0])


if __name__ == "__main__":
    main()