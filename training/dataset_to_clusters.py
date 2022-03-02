"""
Examples

CUDA_VISIBLE_DEVICES=0 python dataset_to_clusters.py --dataset S --s2only

"""
import click
from sklearn import cluster

from train_classifier import *
from manage_settings import get_write_settings, read_models_csv, lazymkdir

from collections import defaultdict

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import pickle

import numpy as np

from tqdm import tqdm

# loading and saving utils for abstract file loading
def _pksave(obj, fname):
    pickle.dump(obj, open(fname, "wb"))

def _pkload(fname):
    return pickle.load(open(fname, "rb"))

def gen_cache_fit(fname, data, genfn, skip = False, loadfn = _pkload, savefn = _pksave):
    if skip:
        obj = genfn(data)
    else:
        if os.path.exists(fname):
            obj = loadfn(fname)
        else:
            obj = genfn(data)
            savefn(obj, fname)
    return obj

def setup_intermed_comp_dir(intermed_comp_dir_base, dataset, biastype = (False, False, False)):
    base = lazymkdir(intermed_comp_dir_base)
    biased, extremebias, s2only = biastype
    if s2only:
        biastype = "s2only"
    elif extremebias:
        biastype = "extreme"
    elif biased:
        biastype = "biased"
    else:
        biastype = "normal"
    # for now I've only implemented test set
    foldername = str(PurePath(base + f"/{dataset}-test-{biastype}/.")) + "/."
    lazymkdir(foldername)
    return str(foldername)


# hack hack hack disgusting
def cuda_dict(tensor_dict):
    for key in tensor_dict.keys():
        tensor_dict[key] = tensor_dict[key].cuda()

## implement each step in the eval, pca, cluster pipeline
# run the ltmodel to get the embeddings
def collect_embeddings(nli_dataset, ltmodel):
    # maybe cuda if we want
    print("Collecting embeddings...")
    for batch in tqdm(nli_dataset.test_dataloader()):
        cuda_dict(batch)
        batch_embs = ltmodel.forward_get_embs(batch)
        yield batch_embs, batch["labels"]

# run the ltmodel to get the posteriors
def collect_posteriors(embs_labs_set_iterator, ltmodel):
    for batch_embs, batch_labs in embs_labs_set_iterator:
        batch_embs = ltmodel.model.classifier.dense(batch_embs[:,0,:])
        batch_embs = batch_embs.squeeze()
        yield batch_embs, batch_labs

# utility fn to convert collected embedding batches into 
def group_by_label(embs_labs_set_iterator):
    label_lists = defaultdict(list)
    for batch_embs, batch_labs in embs_labs_set_iterator:
        if len(batch_embs.shape) == 3:
            for j in range(batch_embs.shape[0]):
                emb = batch_embs[j,0,:].cpu().detach().numpy()
                lab = int(batch_labs[j].cpu().detach().numpy())
                label_lists[lab].append(emb)
        else:
            emb = batch_embs[0,:].cpu().detach().numpy()
            lab = int(batch_labs.cpu().detach().numpy())
            label_lists[lab].append(emb)
    return label_lists

# convert the label_lists dict into bigass numpy arrays
def label_lists_to_arrays(label_lists):
    X_list = []
    labels = []
    for i, key in enumerate(label_lists.keys()):
        X_list += label_lists[key]
        labels += [i] * len(label_lists[key])
    X_list = np.stack(X_list)
    labels = np.array(labels)
    return X_list, labels

# this is a bundle of the two prev functions to interface with auto caching
def get_numpy_embs(nli_data, ltmodel, tmp_save_dir = None):
    if tmp_save_dir == None:
        embs, labs = label_lists_to_arrays(group_by_label(collect_embeddings(nli_data, ltmodel)))
        return embs, labs
    else:
        xname = PurePath(tmp_save_dir + "/embs.npy")
        lname = PurePath(tmp_save_dir + "/labs.npy")
        if os.path.exists(xname) and os.path.exists(lname):
            embs = np.load(xname)
            labs = np.load(lname)
        else:
            embs, labs = label_lists_to_arrays(group_by_label(collect_embeddings(nli_data, ltmodel)))
            np.save(xname, embs)
            np.save(lname, labs)
    return embs, labs

# map the label_lists dict into pca-transformed embeddings
def pca_fit_transform(embs, n_components=50, tmp_save_dir = None):
    print("Performing PCA reduction...")
    if tmp_save_dir == None:
        skip = True
        tmp_save_dir = ""
    else:
        skip = False
    pca_fname = PurePath(tmp_save_dir + f"/pca-{n_components}.pckl")
    pca_model = gen_cache_fit(pca_fname, embs, PCA(n_components=n_components).fit, skip=skip)
    return pca_model.transform(embs)

# produce the clustering
def kmeans_fit_transform(embs, n_clusters=50, tmp_save_dir = None):
    print("Performing KMeans fitting...")    
    if tmp_save_dir == None:
        skip = True
        tmp_save_dir = ""
    else:
        skip = False
    kms_fname = PurePath(tmp_save_dir + f"/kms-{n_clusters}.pckl")
    kms_model = gen_cache_fit(kms_fname, embs, KMeans(n_clusters=n_clusters, init='k-means++').fit, skip=skip)
    return kms_model.predict(embs)


## functions to evaluate and use the clustering; metrics, visualization, etc
# a prereq step is to produce the clustering, from this we can get PECO, etc
def cluster_preds_to_dists(embs_cll, labs, n_clusters):
    cluster_counts = [np.array([0,0,0]) for i in range(n_clusters)]
    for i in range(embs_cll.shape[0]):
        cluster_counts[embs_cll[i]][labs[i]] += 1
    cluster_counts = np.stack(cluster_counts)
    return cluster_counts / np.expand_dims(cluster_counts.sum(-1),1)

# discrete clusterwise cross entropy between local dist and balanced dist
def cluster_xH(cluster_dists, balance = np.array([.333334,.333333,.333333])):
    x = balance * np.log2(cluster_dists)
    return - x.sum(-1)

# compute clusterwise L2 norm between local dist and balanced dist
def cluster_L2(cluster_dists, balance = np.array([.333334,.333333,.333333])):
    x = np.power(balance - cluster_dists, 2)
    return x.sum(-1)

# compute pct of clusters with divergence over some threshold
def threshold_score(cluster_norms, threshold):
    return np.sum(cluster_norms > threshold) / cluster_norms.shape[0]

# get a list of the clusters that are in the set of outliers for a given thresh
def outlier_cluster_ids(cluster_norms, threshold):
    outlier_ids = []
    for i in range(cluster_norms.shape[0]):
        if cluster_norms[i] > threshold:
            outlier_ids.append(i)
    return outlier_ids

# build progressive evaluation of cluster outliers curve
def peco_curve(cluster_norms, n_steps = 20):
    peco_x = []
    peco_y = []
    for i in range(n_steps):
        this_thresh = float(i) / n_steps
        this_count =  threshold_score(cluster_norms, this_thresh)
        peco_x.append(this_thresh)
        peco_y.append(this_count)
    return np.array(peco_x), np.array(peco_y)

# almondo almondo almondo (area under peco curve)
def peco_score(cluster_norms, n_steps = 20):
    _, peco_y = peco_curve(cluster_norms, n_steps)
    # auc
    return np.sum(peco_y) / n_steps

# project a set into tsne for plotting
def tsne_fit_transform(embs, perp, tmp_save_dir = None):
    if tmp_save_dir == None:
        skip = True
        tmp_save_dir = ""
    else:
        skip = False
    tsne_fname = PurePath(tmp_save_dir + f"/tsne-{perp}.np")
    embs_tsne = TSNE(perplexity=perp, verbose=2).fit_transform(embs)
    np.save(tsne_fname, embs_tsne)
    return embs_tsne


@click.command()
@click.option('--n_gpus', default=1, help='number of gpus')
@click.option('--dataset', help="S, M, A1, A2, A3, OC, SICK, etc")
@click.option('--batch_size', default=48)
@click.option('--biased', is_flag=True)
@click.option('--extreme_bias', is_flag=True)
@click.option('--s2only', is_flag=True)
def main(n_gpus, dataset, biased, batch_size, extreme_bias, s2only):
    model_id, pretrained_path = read_models_csv(dataset)
    model, tokenizer = choose_load_model_tokenizer(model_id, dataset)
    ltmodel = RobertaClassifier(model, learning_rate=0)
    print("Init litmodel...")
    if pretrained_path != "":
        ckpt = torch.load(pretrained_path)
        ltmodel.load_state_dict(ckpt["state_dict"])
    else:
        model, tokenizer = choose_load_model_tokenizer(model_id, dataset)
    print("Init dataset...")
    if extreme_bias:
        factor = 0
    else:
        factor = 1

    model.cuda()
    ltmodel.cuda()

    dir_settings = get_write_settings(["data_save_dir", "dataset_dir", "intermed_comp_dir_base"])
    
    intermed_comp_dir = setup_intermed_comp_dir(dir_settings["intermed_comp_dir_base"], dataset,
        (biased, extreme_bias, s2only))

    nli_data = plNLIDataModule(tokenizer, dir_settings["dataset_dir"], dataset, batch_size, biased, factor, s2only)
    nli_data.prepare_data(test_only = True)

    # collect lists of numpy arrays
    embs, labs = get_numpy_embs(nli_data, ltmodel, tmp_save_dir=intermed_comp_dir)
    # pca transformed embeddings
    embs_pca = pca_fit_transform(embs, tmp_save_dir=intermed_comp_dir)
    # cluster-labeled embeddings
    embs_cll = kmeans_fit_transform(embs_pca)
    ## evaluate the PECO measure
    # get the cluster cross entropies
    cluster_dists = cluster_preds_to_dists(embs_cll, labs, n_clusters = 50)
    clusters_xHs = cluster_xH(cluster_dists)
    clusters_L2 = cluster_L2(cluster_dists)
    peco_xH = peco_score(clusters_xHs)
    peco_L2 = peco_score(clusters_L2)
    threshscore_xH_25 = threshold_score(clusters_xHs, .25)
    threshscore_L2_25 = threshold_score(clusters_L2, .25)
    # generate AUC plots
    # generate T-SNE plot
    print("##### REPORT #####")
    lines = ["peco_xH,peco_L2,thresh_xH_25,thresh_L2_25", 
        f"{peco_xH:.4f},{peco_L2:.4f},{threshscore_xH_25:.4f},{threshscore_L2_25:.4f}"]
    with open(PurePath(intermed_comp_dir + "/results.csv"), "w") as f:
        for line in lines:
            print(line)
            f.write(line + "\n")

if __name__ == "__main__":
    main()