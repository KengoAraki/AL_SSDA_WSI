import os
import sys
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import argparse
import glob
import re
import random
import pandas as pd
from tqdm import tqdm

from collections import Counter
import joblib
from natsort import natsorted

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from visualyze import (
    plot_cluster_hist,
    # plot_wsi_distrib_in_all_cl,
    plot_wsi_distrib_in_all_cl_sampled,
    get_embed_feature,
)


def get_wsi_list(facility: str = "MF0003"):
    if facility == "MF0003":
        wsi_list = joblib.load(
            f"/mnt/secssd/AL_SSDA_WSI_strage/dataset/{facility}/trg_l_wsi.jb"
        )
        wsi_list += joblib.load(
            f"/mnt/secssd/AL_SSDA_WSI_strage/dataset/{facility}/trg_unl_wsi.jb"
        )
        wsi_list += joblib.load(
            f"/mnt/secssd/AL_SSDA_WSI_strage/dataset/{facility}/valid_wsi.jb"
        )
    elif facility == "MF0012":
        wsi_list = joblib.load(
            f"/mnt/secssd/AL_SSDA_WSI_strage/dataset/{facility}/cv0_train_{facility}_wsi.jb"
        )
        wsi_list += joblib.load(
            f"/mnt/secssd/AL_SSDA_WSI_strage/dataset/{facility}/cv0_valid_{facility}_wsi.jb"
        )
        wsi_list += joblib.load(
            f"/mnt/secssd/AL_SSDA_WSI_strage/dataset/{facility}/cv0_test_{facility}_wsi.jb"
        )
    else:
        sys.exit(f"{facility} is invalid!")
    return natsorted(wsi_list)


class Clustering(object):
    def __init__(
        self,
        cluster_num: int = 10,
        reduced_dim: int = 30,
        standarize: bool = False,
        random_state: int = 0,
    ):
        self.cluster_num = cluster_num
        self.reduced_dim = reduced_dim
        self.standarize = standarize
        self.random_state = random_state

    # feature_vecsの標準化・次元削減
    def preprocess_feature_vecs(self, feature_vecs: np.ndarray):
        # PCA前に標準化
        if self.standarize:
            feature_vecs = stats.zscore(feature_vecs, axis=0)

        # PCAで次元削減
        pca = PCA(n_components=self.reduced_dim, random_state=self.random_state)
        pca.fit(feature_vecs)
        reduced_feature_vecs = pca.transform(feature_vecs)
        return reduced_feature_vecs

    def run(self, feature_vecs: np.ndarray):
        feature_vecs = self.preprocess_feature_vecs(feature_vecs)

        # KMeansでクラスタリング
        kmeans = KMeans(n_clusters=self.cluster_num, random_state=self.random_state)
        clusters = kmeans.fit(feature_vecs)
        return clusters


# cluster-entropyの算出
def get_cluster_entropy(wsi_cluster_labels: list):
    def calc_entropy(c_val_list: list):
        c_freq_list = [val / sum(c_val_list) for val in c_val_list]
        entropy = stats.entropy(c_freq_list, base=2)
        print(f"entropy: {entropy:.4f}")
        return entropy

    c = Counter(wsi_cluster_labels)
    print(c)
    c_val_list = list(c.values())
    cluster_entropy = calc_entropy(c_val_list)

    return cluster_entropy


# nd_fileが各WSIのクラスごとに分かれている場合
def format_feature_vecs_cl(nd_file_list: list, wsi_list: list = None):
    wsi_idxs = []
    class_labels = []
    wsi_begin_idx, wsi_end_idx = 0, 0
    INIT_FLAG = True

    if wsi_list is None:
        sys.exit("should set wsi_list")
    for wsi in tqdm(wsi_list):
        # 対象のwsiのnd_fileを取得
        wsi_nd_file_list = natsorted(
            list(filter(lambda nd_file, wsi=wsi: f"_{wsi}_" in nd_file, nd_file_list))
        )
        for nd_file in wsi_nd_file_list:
            tmp_feature_vecs = np.load(nd_file)

            if INIT_FLAG:
                feature_vecs = tmp_feature_vecs.copy()
                INIT_FLAG = False
            else:
                feature_vecs = np.concatenate([feature_vecs, tmp_feature_vecs], axis=0)

            m = re.search(r"_cl\d", nd_file)
            cl = int(m.group(0)[-1])
            class_labels.extend(np.full(tmp_feature_vecs.shape[0], cl).tolist())
            wsi_end_idx += tmp_feature_vecs.shape[0]

        # 各wsiのindexを追加
        wsi_idxs.append([wsi_begin_idx, wsi_end_idx])
        wsi_begin_idx = wsi_end_idx

    print(f"feature_vecs: {feature_vecs.shape}")
    print(f"[sample-num] {feature_vecs.shape[0]}")
    return feature_vecs, wsi_idxs, class_labels


def main(args):
    nd_file_list = glob.glob(f"{args.data_dir}*.npy")
    print(len(nd_file_list))

    wsi_list = get_wsi_list(args.facility)
    feature_vecs, wsi_idxs, _ = format_feature_vecs_cl(
        nd_file_list, wsi_list=wsi_list
    )

    clusters = Clustering(
        cluster_num=args.cluster_num,
        reduced_dim=args.reduced_dim,
        standarize=args.standarize,
        random_state=args.random_state,
    ).run(feature_vecs)

    result_dict = {'wsi': [], 'entropy': [], 'sample_num': []}

    for wsi_idx, wsi_name in enumerate(wsi_list):
        print(f"===== {wsi_name} =====")

        idxs = wsi_idxs[wsi_idx]
        wsi_cluster_labels_nd = clusters.labels_[idxs[0]: idxs[1]]
        wsi_cluster_labels = wsi_cluster_labels_nd.tolist()
        cluster_entropy = get_cluster_entropy(wsi_cluster_labels)

        result_dict['wsi'].append(wsi_name)
        result_dict['entropy'].append(f"{cluster_entropy:.6f}")
        result_dict['sample_num'].append(str(len(wsi_cluster_labels)))

        if args.is_hist:
            plot_cluster_hist(
                wsi_cluster_labels_nd,
                cluster_total_num=args.cluster_num,
                title=f"hist_CE_{cluster_entropy:.4f}_{wsi_name}",
                output_dir=args.output_dir
            )

    df = pd.DataFrame(data=result_dict)
    df.to_csv(args.output_dir + f"{args.facility}_cluster_entropy.csv")


def main_distrib_cl(args):
    def random_sample(cl_labels: list, max_k: int = 500):
        """
        cl_labelsから各クラスの要素のindexを取得 → randomにk個サンプリング
        cl_labels (list): feature_vecsの各サンプルのクラスが格納されたリスト
        max_k (int): 各クラスから抽出するサンプル数の上限．足りなければ，全サンプル抽出．
        """
        idxs = []
        cl_labels_nd = np.array(cl_labels)
        for cl in np.unique(cl_labels_nd.tolist()):
            cl_idxs = np.where(cl_labels_nd == cl)[0].tolist()
            if len(cl_idxs) < max_k:
                k = len(cl_idxs)
            else:
                k = max_k
            idxs.extend(random.sample(cl_idxs, k=k))
        return natsorted(idxs)

    nd_file_list = glob.glob(f"{args.data_dir}*.npy")
    print(len(nd_file_list))

    wsi_list = get_wsi_list(args.facility)
    feature_vecs, wsi_idxs, cl_labels = format_feature_vecs_cl(
        nd_file_list, wsi_list=wsi_list
    )

    print("run clustering...")
    clusters = Clustering(
        cluster_num=args.cluster_num,
        reduced_dim=args.reduced_dim,
        standarize=args.standarize,
        random_state=args.random_state,
    ).run(feature_vecs)

    print("run tsne...")
    x_embedded = get_embed_feature(feature_vecs, method="tsne")
    # np.save(
    #     f"{args.output_dir}x_embedded", x_embedded,
    # )
    # x_embedded = np.load(
    #     "/home/kengoaraki/Project/DA/SSDA_Annot_WSI/analysis/output/cluster_distrib/x_embedded.npy"
    # )

    # cl_labelsから各クラスの要素のindexを取得→randomにk個サンプリング
    sampled_x_idxs = random_sample(cl_labels, max_k=500)

    for wsi_idx, wsi_name in enumerate(wsi_list):
        print(f"===== {wsi_name} =====")

        trg_idxs = wsi_idxs[wsi_idx]
        trg_x_idxs = list(range(trg_idxs[0], trg_idxs[1]))
        ele_num = trg_idxs[1] - trg_idxs[0]

        idxs = wsi_idxs[wsi_idx]
        wsi_cluster_labels_nd = clusters.labels_[idxs[0]: idxs[1]]
        wsi_cluster_labels = wsi_cluster_labels_nd.tolist()
        cluster_entropy = get_cluster_entropy(wsi_cluster_labels)

        # plot_wsi_distrib_in_all_cl(
        #     feature_vecs,
        #     wsi_x_idxs=trg_idxs,
        #     cluster_labels=clusters.labels_,
        #     cl_labels=cl_labels,
        #     method="tsne",
        #     output_dir=args.output_dir,
        #     title=f"{wsi_name}_ele{ele_num}",
        # )

        plot_wsi_distrib_in_all_cl_sampled(
            x_embedded,
            sampled_x_idxs=sampled_x_idxs,
            wsi_x_idxs=trg_x_idxs,
            cluster_labels=clusters.labels_,
            cl_labels=cl_labels,
            method="tsne",
            output_dir=args.output_dir,
            title=f"tsne_CE_{cluster_entropy:.4f}_ele{ele_num}_{wsi_name}",
        )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reduced_dim", type=int, default=30)
    parser.add_argument("--cluster_num", type=int, default=10)
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument(
        "--standarize",
        type=bool,
        default=False,
        help="standarize feature vectors before PCA",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="/mnt/secssd/AL_SSDA_WSI_MICCAI_srcMF0003_strage/cluster_entropy/MF0012/npy/",
    )
    parser.add_argument("--facility", type=str, default="MF0012")
    parser.add_argument(
        "--output_dir", type=str, default="/mnt/secssd/AL_SSDA_WSI_MICCAI_srcMF0003_strage/cluster_entropy/MF0012/"
    )
    parser.add_argument("--is_hist", type=bool, default=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    args = get_args()

    # fix seed
    random.seed(args.random_state)
    np.random.seed(args.random_state)

    # main(args)
    main_distrib_cl(args)
