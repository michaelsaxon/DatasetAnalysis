"""
CUDA_VISIBLE_DEVICES=3 python compare_clusts.py --dataset M
"""

from train_classifier import *
from dataset_to_clusters import *
from compare_runs import *
from manage_settings import get_write_settings, read_models_csv, lazymkdir
from tqdm import tqdm

def get_numpy_embs_clls(nli_data, ltmodel, n_clusters, intermed_comp_dir = "", lastdense = False):
    # collect lists of numpy arrays
    embs, labs = get_numpy_embs(nli_data, ltmodel, tmp_save_dir=intermed_comp_dir, lastdense = lastdense)
    # pca transformed embeddings
    embs_pca = pca_fit_transform(embs, tmp_save_dir=intermed_comp_dir)
    # cluster-labeled embeddings
    embs_cll = kmeans_fit_transform(embs_pca, n_clusters = n_clusters)
    return embs, labs, embs_cll


def single_dataset_experiment(dataset, batch_size, s1only, n_clusters, lastdense):
    # do 2 diff pretrained paths
    model_id, pretrained_path_1 = read_models_csv(dataset)
    _, pretrained_path_2 = read_models_csv(dataset, s2only=not s1only, s1only = s1only)

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

    intermed_comp_dir_1 = setup_intermed_comp_dir(dir_settings["intermed_comp_dir_base"], dataset,
        n_clusters, lastdense, (False, False, False or s1only))

    embs_1, labs_1, clls_1 = get_numpy_embs_clls(nli_data_1, ltmodel, n_clusters, intermed_comp_dir_1)
    vects_1 = get_cluster_vectors(embs_1, clls_1)

    print("getting ds 2 on model 1...")

    intermed_comp_dir_3 = setup_intermed_comp_dir(dir_settings["intermed_comp_dir_base"], dataset,
        n_clusters, lastdense, (True, False, True or s1only))

    intermed_comp_dir_3 = str(PurePath(intermed_comp_dir_3[:-2] + "-on_normal/."))
    embs_2, _ = get_numpy_embs(nli_data_2, ltmodel, tmp_save_dir=intermed_comp_dir_3, lastdense = lastdense)

    ltmodel.load_state_dict(ckpt_2["state_dict"])
    model.cuda()
    ltmodel.cuda()

    print("Running model 2...")

    intermed_comp_dir_2 = setup_intermed_comp_dir(dir_settings["intermed_comp_dir_base"], dataset,
        n_clusters, lastdense, (True, False, True or s1only))

    _, labs_2, clls_2 = get_numpy_embs_clls(nli_data_2, ltmodel, n_clusters, intermed_comp_dir_2)

    vects_2 = get_cluster_vectors(embs_2, clls_2)

    return greedy_cluster_meanings_comparison(vects_1, vects_2)


@click.command()
@click.option('--dataset', help="S, M, A1, A2, A3, OC, SICK, etc")
@click.option('--batch_size', default=16)
@click.option('--s1only', is_flag=True)
@click.option('--n_clusters', default=50)
@click.option('--lastdense', is_flag=True)
def main(dataset, batch_size, s1only, n_clusters, lastdense):
    print(single_dataset_experiment(dataset, batch_size, s1only, n_clusters, lastdense))


if __name__ == "__main__":
    #main()
    results = []
    for ds in ["SICK", "S", "MB", "MU", "X", "F", "AA", "A1", "A2", "A3", "OC", "CF"]:
        if ds == "F":
            results.append(single_dataset_experiment(ds, 8, True, 50, False))
        elif ds == "CF":
            results.append(single_dataset_experiment(ds, 16, True, 50, False))
            results.append(single_dataset_experiment(ds, 16, False, 50, False))
        else:
            results.append(single_dataset_experiment(ds, 16, False, 50, False))
    print("########FINAL RESULTS#########")
    print("avg_cossim, cossim@.8, cossim@0")
    for r1, r2, r3 in results:
        print(f"{r1:.4f}, {r2:.4f}, {r3:.4f}")
        
