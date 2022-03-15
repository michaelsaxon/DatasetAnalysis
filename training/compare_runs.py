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
            grad_outputs = torch.ones_like(batch_posts.logits))[0][:,1:]
        importance_map = torch.norm(local_grad, dim=2)
        local_importance_maps = importance_map / torch.sum(importance_map, dim=-1).unsqueeze(1)
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

def padcombo(list_of_2darrays):
    length = max(map(lambda x: x.shape[1], list_of_2darrays))
    return np.concatenate(
        list(map(lambda x: np.pad(x, ((0,0),(length-x.shape[1],0))), list_of_2darrays)),
        0)


def get_numpy_preds_imp_maps(nli_data, ltmodel):
    imp_maps_list = []
    decisions_list = []
    labels_list = []
    for b_impmaps, b_posts, b_labs in collect_importance_maps_and_posteriors(nli_data, ltmodel):
        batch_impmaps = b_impmaps.cpu().detach().numpy()
        imp_maps_list.append(batch_impmaps)
        batch_decisions = torch.max(b_posts, -1).indices
        batch_decisions = batch_decisions.cpu().detach().numpy()
        #print(batch_decisions.shape)
        batch_labs = b_labs.cpu().detach().numpy()
        decisions_list.append(batch_decisions)
        labels_list.append(batch_labs)
    imp_maps_list = padcombo(imp_maps_list)
    decisions_list = np.concatenate(decisions_list)
    labels_list = np.concatenate(labels_list)
    return decisions_list, labels_list, imp_maps_list


def cosine_sim(mat_1, mat_2):
    num = mat_1 * mat_2
    denom_1 = mat_1 * mat_1
    denom_2 = mat_2 * mat_2
    num =  num.sum(-1) / (np.sqrt(denom_1.sum(-1)) * np.sqrt(denom_2.sum(-1)))
    return num.mean()


def row_agreements(maps_1, maps_2):
    print(maps_2.shape[-1])
    # cosine sim
    return cosine_sim(maps_1, maps_2)


def get_top_n_by_row(mat, n):
    m1_top_10 = np.argsort(mat)[:,-n]
    tops = []
    for i in range(mat.shape[0]):
        tops.append(mat[i,m1_top_10[i]])
    tops = np.expand_dims(np.array(tops),1)
    out = mat * (mat > tops)
    return out / np.expand_dims(out.sum(-1),1)


@click.command()
@click.option('--n_gpus', default=1, help='number of gpus')
@click.option('--dataset', help="S, M, A1, A2, A3, OC, SICK, etc")
@click.option('--batch_size', default=48)
@click.option('--top_n', default=10)
def main(n_gpus, dataset, batch_size, top_n):
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
    total_size = max(maps_1.shape[1], maps_2.shape[1])
    maps_1 = np.pad(maps_1, ((0,0),(total_size-maps_1.shape[1],0)))
    maps_2 = np.pad(maps_2, ((0,0),(total_size-maps_2.shape[1],0)))

    # align maps_1 and maps_2
    map_agreement = row_agreements(maps_1, maps_2)
    #print(map_agreement)
    # how much of the attention weight is in s2 for regular condition
    s2_attn_full = np.not_equal(maps_2, 0) * maps_1
    s2_attn_full = s2_attn_full.sum(-1).squeeze()

    maps_1_top = get_top_n_by_row(maps_1, top_n)
    maps_2_top = get_top_n_by_row(maps_2, top_n)
    map_top_agreement = row_agreements(maps_1_top, maps_2_top)


    labs_1 = labs_1.squeeze()
    labs_2 = labs_2.squeeze()
    agreement = np.equal(dec_1, dec_2)
    correct_1 = np.equal(dec_1, labs_1)
    correct_2 = np.equal(dec_2, labs_2)
    correct_agreement = agreement * correct_1 * correct_2
    correct_1 + correct_2

    '''
    print(agreement.sum() / agreement.shape[0])
    print(correct_agreement.sum() / correct_1.sum())
    print(correct_agreement.sum() / correct_2.sum())
    print(correct_agreement.sum() / agreement.sum())
    print("map agreement lmao")
    print(map_agreement.mean())
    print(map_top_agreement.mean())
    print(s2_attn_full.mean())
    print(" with agreement ")
    print(np.sum(map_agreement * agreement) / agreement.sum())
    print(np.sum(map_top_agreement * agreement) / agreement.sum())
    print(np.sum(s2_attn_full * agreement) / agreement.sum())
    '''
    print(f"{agreement.sum() / agreement.shape[0]*100:.2f},{agreement.sum() / correct_1.sum()*100:.2f},{map_agreement.mean()}")


if __name__ == "__main__":
    main()