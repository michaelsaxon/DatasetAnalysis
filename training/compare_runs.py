"""
CUDA_VISIBLE_DEVICES=0 python compare_runs.py --dataset A1
"""

from train_classifier import *
from dataset_to_clusters import *
from manage_settings import get_write_settings, read_models_csv, lazymkdir
from tqdm import tqdm

def collect_posteriors(nli_dataset, ltmodel):
    for batch in tqdm(nli_dataset.test_dataloader()):
        cuda_dict(batch)
        batch_posts = ltmodel(input_ids = batch['input_ids'], attention_mask=batch['attention_mask'])
        yield batch_posts.logits, batch["labels"]

def collect_importance_maps_and_posteriors(nli_dataset, ltmodel):
    for batch in tqdm(nli_dataset.test_dataloader()):
        cuda_dict(batch)
        batch_posts = ltmodel(input_ids = batch['input_ids'], attention_mask=batch['attention_mask'], 
            output_hidden_states = True)
        local_grad = torch.autograd.grad(batch_posts.logits,
            batch_posts.hidden_states[0], retain_graph = True,
            grad_outputs = torch.ones_like(batch_posts.logits))[0]
        local_importance_maps = torch.norm(local_grad, dim=2) / torch.sum(torch.norm(local_grad, dim=2))
        yield local_importance_maps, batch_posts.logits, batch["labels"]


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


def get_numpy_preds_imp_maps(nli_data, ltmodel):
    imp_maps_list = []
    decisions_list = []
    labels_list = []
    for b_impmaps, b_posts, b_labs in collect_importance_maps_and_posteriors(nli_data, ltmodel):
        print(batch_impmaps.shape)
        batch_impmaps = b_impmaps.cpu().detach().numpy()
        imp_maps_list.append(batch_impmaps)
        batch_decisions = torch.max(b_posts, -1).indices
        batch_decisions = batch_decisions.cpu().detach().numpy()
        print(batch_decisions.shape)
        batch_labs = b_labs.cpu().detach().numpy()
        decisions_list.append(batch_decisions)
        labels_list.append(batch_labs)
    imp_maps_list = np.concatenate(imp_maps_list)
    decisions_list = np.concatenate(decisions_list)
    labels_list = np.concatenate(labels_list)
    return decisions_list, labels_list, imp_maps_list


@click.command()
@click.option('--n_gpus', default=1, help='number of gpus')
@click.option('--dataset', help="S, M, A1, A2, A3, OC, SICK, etc")
@click.option('--batch_size', default=48)
def main(n_gpus, dataset, batch_size):
    # do 2 diff pretrained paths
    model_id, pretrained_path_1 = read_models_csv(dataset)
    _, pretrained_path_2 = read_models_csv(dataset, s2only=True)

    model, tokenizer = choose_load_model_tokenizer(model_id, dataset)
    ltmodel = RobertaClassifier(model, learning_rate=0)
    print("Init litmodel...")
    ckpt_1 = torch.load(pretrained_path_1)
    ckpt_2 = torch.load(pretrained_path_2)
    ltmodel.load_state_dict(ckpt_1["state_dict"])
    print("Init dataset...")

    model.cuda()
    ltmodel.cuda()

    dir_settings = get_write_settings(["data_save_dir", "dataset_dir", "intermed_comp_dir_base"])
    
    nli_data_1 = plNLIDataModule(tokenizer, dir_settings["dataset_dir"], dataset, batch_size, False, 1, False)
    nli_data_1.prepare_data(test_only = True)

    nli_data_2 = plNLIDataModule(tokenizer, dir_settings["dataset_dir"], dataset, batch_size, False, 1, True)
    nli_data_2.prepare_data(test_only = True)

    print("Running model 1...")

    dec_1, labs_1, maps_1 = get_numpy_preds_imp_maps(nli_data_1, ltmodel)

    ltmodel.load_state_dict(ckpt_2["state_dict"])
    model.cuda()
    ltmodel.cuda()

    print("Running model 2...")

    dec_2, labs_2, maps_2 = get_numpy_preds_imp_maps(nli_data_2, ltmodel)

    labs_1 = labs_1.squeeze()
    labs_2 = labs_2.squeeze()
    agreement = np.equal(dec_1, dec_2)
    correct_1 = np.equal(dec_1, labs_1)
    correct_2 = np.equal(dec_2, labs_2)
    correct_agreement = agreement * correct_1 * correct_2
    correct_1 + correct_2

    print(agreement.sum() / agreement.shape[0])
    print(correct_agreement.sum() / correct_1.sum())
    print(correct_agreement.sum() / correct_2.sum())
    print(correct_agreement.sum() / agreement.sum())


if __name__ == "__main__":
    main()