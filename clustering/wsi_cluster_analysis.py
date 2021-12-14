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

from collections import Counter
import joblib
from natsort import natsorted

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from visualyze import (
    plot_cluster_hist,
    # plot_wsi_distrib_in_all_cl,
    plot_wsi_distrib_in_all_cl_sampled,
)


def get_wsi_list(jb_dir: str, facility: str = "MF0003", cv_num: int = 0):
    wsi_list = joblib.load(
        jb_dir + f"{facility}/" + f"cv{cv_num}_train_" + f"{facility}_wsi.jb"
    )
    wsi_list += joblib.load(
        jb_dir + f"{facility}/" + f"cv{cv_num}_valid_" + f"{facility}_wsi.jb"
    )
    wsi_list += joblib.load(
        jb_dir + f"{facility}/" + f"cv{cv_num}_test_" + f"{facility}_wsi.jb"
    )
    return natsorted(wsi_list)


# nd_fileが各WSIのクラスごとに分かれていない場合
def format_feature_vecs(nd_file_list: list):
    wsi_idxs = []
    begin_idx, end_idx = 0, 0
    INIT_FLAG = True

    for nd_file in nd_file_list:
        tmp_feature_vecs = np.load(nd_file)
        if INIT_FLAG:
            feature_vecs = tmp_feature_vecs.copy()
            INIT_FLAG = False
        else:
            feature_vecs = np.concatenate([feature_vecs, tmp_feature_vecs], axis=0)

        # 各wsiのindexを追加
        end_idx += tmp_feature_vecs.shape[0]
        wsi_idxs.append([begin_idx, end_idx])
        begin_idx = end_idx

    print(f"feature_vecs: {feature_vecs.shape}")
    print(f"[sample-num] {feature_vecs.shape[0]}")
    return feature_vecs, wsi_idxs


# nd_fileが各WSIのクラスごとに分かれている場合
def format_feature_vecs_cl(nd_file_list: list, wsi_list: list = None):
    wsi_idxs = []
    class_labels = []
    wsi_begin_idx, wsi_end_idx = 0, 0
    INIT_FLAG = True

    if wsi_list is None:
        sys.exit("should set wsi_list")
    for wsi in wsi_list:
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


# feature_vecsの標準化・次元削減
def preprocess_feature_vecs(
    feature_vecs,
    reduced_dim: int = 30,
    random_state: int = 0,
    standarize: bool = False,
):
    # PCA前に標準化
    if standarize:
        feature_vecs = stats.zscore(feature_vecs, axis=0)

    # PCAで次元削減
    pca = PCA(n_components=reduced_dim, random_state=random_state)
    pca.fit(feature_vecs)
    reduced_feature_vecs = pca.transform(feature_vecs)
    return reduced_feature_vecs


# 2クラス用
def get_cluster_data(
    nd_feature_vecs, x_cluster_labels,
):
    feature_vecs_list = []
    for i in range(x_cluster_labels.max() + 1):
        # cluster-iのラベルを抽出
        i_cluster_labels = x_cluster_labels == i
        # cluster-iのfeature_vecs
        i_cluster_feature_vecs = nd_feature_vecs[i_cluster_labels, :]
        feature_vecs_list.append(i_cluster_feature_vecs)
    return feature_vecs_list


def calc_entropy(c_val_list: list):
    c_freq_list = [val / sum(c_val_list) for val in c_val_list]
    entropy = stats.entropy(c_freq_list, base=2)
    print(f"entropy: {entropy:.4f}")
    return entropy


def main_hist(args):
    nd_file_list = glob.glob(f"{args.data_dir}{args.facility}/*.npy")
    print(len(nd_file_list))

    feature_vecs, wsi_idxs = format_feature_vecs(nd_file_list)
    feature_vecs = preprocess_feature_vecs(
        feature_vecs,
        reduced_dim=args.reduced_dim,
        random_state=args.random_state,
        standarize=args.standarize,
    )

    # KMeansでクラスタリング
    kmeans = KMeans(n_clusters=args.cluster_num, random_state=args.random_state)
    clusters = kmeans.fit(feature_vecs)

    # # clusterごとのfeature_vecsを取得
    # feature_vecs_list = get_cluster_data(
    #     feature_vecs, clusters.labels_)

    # for cluster_idx in range(len(feature_vecs_list)):
    #     i_feature_vecs = feature_vecs_list[cluster_idx]

    wsi_nums = len(wsi_idxs)
    for wsi_idx in range(wsi_nums):
        wsi_name = os.path.splitext(os.path.basename(nd_file_list[wsi_idx]))[0]
        print(f"[{wsi_name}]")

        idxs = wsi_idxs[wsi_idx]

        wsi_clusters = clusters.labels_[idxs[0]: idxs[1]]
        c = Counter(wsi_clusters.tolist())
        print(c)
        c_val_list = list(c.values())
        entropy = calc_entropy(c_val_list)
        wsi_name_short = wsi_name.replace("srcMF0012_all_cv0_MF0003_", "")
        plot_cluster_hist(
            wsi_clusters,
            cluster_total_num=args.cluster_num,
            title=f"Hist_ClusterID-Freq_Entropy{entropy:.4f}_{wsi_name_short}",
            output_dir=args.output_dir,
            subtitle=f"{c.most_common()}",
        )


def main_hist_cl(args):
    nd_file_list = glob.glob(f"{args.data_dir}{args.facility}/*.npy")
    print(len(nd_file_list))

    wsi_list = get_wsi_list(args.jb_dir, facility=args.facility, cv_num=args.cv_num)
    feature_vecs, wsi_idxs, cl_labels = format_feature_vecs_cl(
        nd_file_list, wsi_list=wsi_list
    )

    feature_vecs = preprocess_feature_vecs(
        feature_vecs,
        reduced_dim=args.reduced_dim,
        random_state=args.random_state,
        standarize=args.standarize,
    )

    # KMeansでクラスタリング
    kmeans = KMeans(n_clusters=args.cluster_num, random_state=args.random_state)
    clusters = kmeans.fit(feature_vecs)

    for wsi_idx, wsi_name in enumerate(wsi_list):
        print(f"[{wsi_name}]")

        idxs = wsi_idxs[wsi_idx]
        wsi_clusters = clusters.labels_[idxs[0]: idxs[1]]
        c = Counter(wsi_clusters.tolist())
        print(c)
        c_val_list = list(c.values())
        entropy = calc_entropy(c_val_list)
        plot_cluster_hist(
            wsi_clusters,
            cluster_total_num=args.cluster_num,
            title=f"Hist_ClusterID-Freq_Entropy{entropy:.4f}_{wsi_name}",
            output_dir=args.output_dir,
            subtitle=f"{c.most_common()}",
        )


def main_distrib_cl(args):
    def random_sample(cl_labels: list, max_k: int = 400):
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

    nd_file_list = glob.glob(f"{args.data_dir}{args.facility}/*.npy")
    print(len(nd_file_list))

    wsi_list = get_wsi_list(args.jb_dir, facility=args.facility, cv_num=args.cv_num)
    feature_vecs, wsi_idxs, cl_labels = format_feature_vecs_cl(
        nd_file_list, wsi_list=wsi_list
    )
    feature_vecs = preprocess_feature_vecs(
        feature_vecs,
        reduced_dim=args.reduced_dim,
        random_state=args.random_state,
        standarize=args.standarize,
    )

    # KMeansでクラスタリング
    kmeans = KMeans(n_clusters=args.cluster_num, random_state=args.random_state)
    clusters = kmeans.fit(feature_vecs)
    cluster_labels = clusters.labels_

    # from visualyze import get_embed_feature

    # x_embedded = get_embed_feature(feature_vecs, method="tsne")
    # np.save(
    #     f"{args.output_dir}x_embedded", x_embedded,
    # )

    x_embedded = np.load(
        "/home/kengoaraki/Project/DA/SSDA_Annot_WSI/analysis/output/cluster_distrib/x_embedded.npy"
    )

    # cl_labelsから各クラスの要素のindexを取得→randomにk個サンプリング
    sampled_x_idxs = random_sample(cl_labels, max_k=400)

    for wsi_idx, wsi_name in enumerate(wsi_list):
        print(f"[{wsi_name}]")

        trg_idxs = wsi_idxs[wsi_idx]
        trg_x_idxs = list(range(trg_idxs[0], trg_idxs[1]))
        ele_num = trg_idxs[1] - trg_idxs[0]

        # plot_wsi_distrib_in_all_cl(
        #     feature_vecs,
        #     wsi_x_idxs=trg_idxs,
        #     cluster_labels=cluster_labels,
        #     cl_labels=cl_labels,
        #     method="tsne",
        #     output_dir=args.output_dir,
        #     title=f"{wsi_name}_ele{ele_num}",
        # )

        plot_wsi_distrib_in_all_cl_sampled(
            x_embedded,
            sampled_x_idxs=sampled_x_idxs,
            wsi_x_idxs=trg_x_idxs,
            cluster_labels=cluster_labels,
            cl_labels=cl_labels,
            method="tsne",
            output_dir=args.output_dir,
            title=f"{wsi_name}_ele{ele_num}",
        )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/mnt/secssd/SSDA_Annot_WSI_strage/analysis/feature_data_wsi/",
    )
    # parser.add_argument(
    #     "--output_dir", type=str, default="./analysis/output/cluster_hist/"
    # )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/kengoaraki/Project/DA/SSDA_Annot_WSI/analysis/output/cluster_distrib/",
    )
    parser.add_argument("--facility", type=str, default="MF0003")
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
        "--jb_dir", type=str, default="/mnt/secssd/SSDA_Annot_WSI_strage/dataset/"
    )
    parser.add_argument("--cv_num", type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    # fix seed
    random.seed(args.random_state)
    np.random.seed(args.random_state)

    # main_hist_cl(args)
    main_distrib_cl(args)
