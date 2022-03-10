import os
import sys
import logging
import matplotlib.pyplot as plt
import time
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from plot_miccai.vis_utils import markers, pltcolor, set_colors
from plot_miccai.feature import get_embed_feature


def merge_vecs_list(
    x_list: list,
    x_all: list = [],
    cl_idx_list: list = [],
    num_x_list: list = [],
    set_idx_list: list = [],
    set_idx: int = 0
):
    for cl_idx, x in enumerate(x_list):
        cl_idx_list.append(cl_idx)
        num_x_list.append(x.shape[0])
        set_idx_list.append(set_idx)
        if len(x_all) < 1:
            x_all = x
        else:
            if x.shape[0] > 0:  # このクラスのdataが存在しない場合はスキップ
                x_all = np.concatenate([x_all, x], axis=0)
    return x_all, cl_idx_list, num_x_list, set_idx_list


# labeled_src, labeled_trg, unlebeled_trgをplot
def plot_feature_space_miccai(
    src_l_vecs_list: list,
    trg_l_vecs_list: list,
    trg_unl_vecs_list: list,
    other_vecs_list: list,
    method: str = "tsne",
    output_dir: str = None,
    title: str = "feature_space",
    sample_N: int = 500,
):
    start_time = time.time()

    print(f"Visualize features with {method}")
    plt.figure(figsize=(9, 6))
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    set_dict = {"src_l": 0, "trg_l": 1, "trg_unl": 2}

    # 各特徴量をマージ
    x_all, cl_idx_list, num_x_list, set_idx_list = \
        merge_vecs_list(src_l_vecs_list, set_idx=set_dict['src_l'])
    x_all, cl_idx_list, num_x_list, set_idx_list = \
        merge_vecs_list(trg_unl_vecs_list, x_all, cl_idx_list, num_x_list, set_idx_list, set_idx=set_dict['trg_unl'])
    x_all, cl_idx_list, num_x_list, set_idx_list = \
        merge_vecs_list(trg_l_vecs_list, x_all, cl_idx_list, num_x_list, set_idx_list, set_idx=set_dict['trg_l'])

    # 表示しないがt-SNEの次元圧縮時に考慮すべきベクトル
    for cl_idx, x in enumerate(other_vecs_list):
        x_all = np.concatenate([x_all, x], axis=0)

    # 次元圧縮
    x_embedded = get_embed_feature(x_all, method)
    np.savez(f"{output_dir}{title}_{method}", x_embedded)

    # x_embedded = np.load(f"{output_dir}{title}_{method}.npz", allow_pickle=True)['arr_0']

    # total_num = 0
    # for i, (cl_idx, num_x, set_idx) in enumerate(
    #     zip(cl_idx_list, num_x_list, set_idx_list)
    # ):
    #     # データが存在しない場合はスキップ
    #     if num_x < 1:
    #         continue

    #     # color (set別)
    #     c = [pltcolor(label=set_idx, cols=set_colors)]
    #     # marker (class別)
    #     marker = markers[cl_idx]

    #     plt.scatter(
    #         x_embedded[total_num: total_num + num_x, 0],
    #         x_embedded[total_num: total_num + num_x, 1],
    #         c=c,
    #         marker=marker,
    #         label=None,
    #         alpha=0.8,
    #         linewidth=0,
    #         edgecolors='face'
    #     )
    #     total_num += num_x

    total_num = 0
    for i, (cl_idx, num_x, set_idx) in enumerate(
        zip(cl_idx_list, num_x_list, set_idx_list)
    ):
        # データが存在しない場合はスキップ
        if num_x < 1:
            continue

        # color (set別)
        c = [pltcolor(label=set_idx, cols=set_colors)]
        # marker (class別)
        marker = markers[cl_idx]

        if sample_N < num_x:
            N = sample_N
        else:
            N = num_x

        plt.scatter(
            x_embedded[total_num: total_num + N, 0],
            x_embedded[total_num: total_num + N, 1],
            c=c,
            s=10,
            marker=marker,
            label=None,
            alpha=0.8,
            linewidth=0,
            edgecolors='face'
        )
        total_num += num_x

    # 凡例用 (class)
    plt.scatter([], [], c="silver", alpha=1, marker=markers[0], label="Non-Neop")
    plt.scatter([], [], c="silver", alpha=1, marker=markers[2], label="LSIL")
    plt.scatter([], [], c="silver", alpha=1, marker=markers[1], label="HSIL")
    # 凡例用 (set)
    c = [pltcolor(label=set_dict['src_l'], cols=set_colors)]
    plt.scatter([], [], c=c, alpha=1, marker=markers[0], label="Labeled Source")
    c = [pltcolor(label=set_dict['trg_l'], cols=set_colors)]
    plt.scatter([], [], c=c, alpha=1, marker=markers[0], label="Labeled Target")
    c = [pltcolor(label=set_dict['trg_unl'], cols=set_colors)]
    plt.scatter([], [], c=c, alpha=1, marker=markers[0], label="Unlabeled Target")

    elapsed_time = time.time() - start_time
    logging.info(f"elapsed_time: {elapsed_time:.1f} [sec]")

    plt.title(f"{title} with {method}")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    if output_dir is not None:
        plt.savefig(
            f"{output_dir}{title}_{method}.png",
            format="png",
            dpi=500,
            bbox_inches="tight",
        )
    # plt.show()
    plt.clf()
    plt.close()
