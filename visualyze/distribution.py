import os
import sys
import logging
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.decomposition import PCA
from umap import UMAP

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from visualyze.utils import edgecolor, imscatter, colors_domains, markers


def get_embed_feature(x, method: str):
    def tsne_embed(x):
        tsne = TSNE(n_components=2, random_state=0, perplexity=30)
        x_embedded = tsne.fit_transform(x)
        return x_embedded

    def pca_embed(x):
        pca = PCA(n_components=2, random_state=0)
        x_embedded = pca.fit_transform(x)
        return x_embedded

    def umap_embed(x):
        umap = UMAP(n_components=2, random_state=0, n_neighbors=50)
        x_embedded = umap.fit_transform(x)
        return x_embedded

    if method == "tsne":
        x_embedded = tsne_embed(x)
    elif method == "pca":
        x_embedded = pca_embed(x)
    elif method == "umap":
        x_embedded = umap_embed(x)
    elif method == "direct":
        x_embedded = x
    else:
        sys.exit(f"cannot find method: {method}")
    return x_embedded


def plot_feature_space(x, labels, colormap, method="tsne"):
    f"Visualize features with {method}"
    plt.figure(figsize=(8, 6))

    x_embedded = get_embed_feature(x, method)

    plt.scatter(x_embedded[:, 0], x_embedded[:, 1], c=labels, cmap=colormap)
    # plt.colorbar()

    plt.title(f"Embedding Space with {method}")
    plt.show()
    plt.clf()
    plt.close()


def plot_two_dist(
    x1, x2, method: str = "tsne", output_dir: str = None, title: str = "feature_space"
):
    print(f"Plot feature space with {method}")
    plt.figure(figsize=(8, 6))

    start_time = time.time()
    x_all = np.concatenate([x1, x2], axis=0)
    num_x_list = [x1.shape[0], x2.shape[0]]

    x_embedded = get_embed_feature(x_all, method)

    cmap = plt.get_cmap("tab10")
    for i, num_x in enumerate(num_x_list):
        plt.scatter(
            x_embedded[i * num_x: (i * num_x) + num_x, 0],
            x_embedded[i * num_x: (i * num_x) + num_x, 1],
            c=[cmap(i)],
        )

    elapsed_time = time.time() - start_time
    logging.info(f"elapsed_time({method}): {elapsed_time:.1f} [sec]")

    plt.title(f"Embedding space with {method}")
    if output_dir is not None:
        plt.savefig(f"{output_dir}{title}_{method}.png", format="png", dpi=300)
    # plt.show()
    plt.clf()
    plt.close()


def plot_wsi_distrib_in_all(
    x_all,
    wsi_x_idxs: list,
    method: str = "tsne",
    output_dir: str = None,
    title: str = "feature_space",
):
    print(f"Plot feature space with {method}")
    plt.figure(figsize=(8, 6))

    start_time = time.time()
    x_embedded = get_embed_feature(x_all, method)

    cmap = plt.get_cmap("tab10")
    plt.scatter(x_embedded[:, 0], x_embedded[:, 1], c=[cmap(0)])
    plt.scatter(
        x_embedded[wsi_x_idxs[0]: wsi_x_idxs[1], 0],
        x_embedded[wsi_x_idxs[0]: wsi_x_idxs[1], 1],
        c=[cmap(1)],
    )

    elapsed_time = time.time() - start_time
    logging.info(f"elapsed_time({method}): {elapsed_time:.1f} [sec]")

    plt.title(f"{title} with {method}")
    if output_dir is not None:
        plt.savefig(f"{output_dir}{title}_{method}.png", format="png", dpi=300)
    # plt.show()
    plt.clf()
    plt.close()


# def plot_wsi_distrib_in_all_cl(
#     feature_vecs,
#     wsi_x_idxs: list,
#     cluster_labels: list = None,
#     cl_labels: list = None,
#     method: str = "tsne",
#     output_dir: str = None,
#     title: str = "feature_space",
#     markers: list = markers,
# ):
#     print(f"Plot feature space with {method}")
#     plt.figure(figsize=(8, 6))
#     cmap = plt.get_cmap("tab20")

#     start_time = time.time()
#     x_embedded = get_embed_feature(feature_vecs, method)

#     for i, (cluster_label, cl_label) in enumerate(zip(tqdm(cluster_labels), cl_labels)):
#         c = [cmap(cluster_label)]
#         marker = markers[cl_label]
#         label = f"cluster{cluster_label:02d}_cl{cl_label}"

#         if (i >= wsi_x_idxs[0]) and (i < wsi_x_idxs[1]):
#             alpha = 1
#             linewidth = 3
#         else:
#             alpha = 0.7
#             linewidth = 1.5

#         plt.scatter(
#             x_embedded[i, 0],
#             x_embedded[i, 1],
#             c=c,
#             marker=marker,
#             label=label,
#             alpha=alpha,
#             linewidth=linewidth,
#         )

#     elapsed_time = time.time() - start_time
#     logging.info(f"elapsed_time({method}): {elapsed_time:.1f} [sec]")

#     plt.title(f"{title} with {method}")
#     if output_dir is not None:
#         plt.savefig(f"{output_dir}{title}_{method}.png", format="png", dpi=300)
#     # plt.show()
#     plt.clf()
#     plt.close()


def plot_wsi_distrib_in_all_cl(
    x_embedded,
    wsi_x_idxs: list,
    cluster_labels: list = None,
    cl_labels: list = None,
    method: str = "tsne",
    output_dir: str = None,
    title: str = "feature_space",
    markers: list = markers,
):
    print(f"Plot feature space with {method}")
    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap("tab20")

    start_time = time.time()

    for i, (cluster_label, cl_label) in enumerate(zip(tqdm(cluster_labels), cl_labels)):
        c = [cmap(cluster_label)]
        marker = markers[cl_label]
        label = f"cluster{cluster_label:02d}_cl{cl_label}"

        if (i >= wsi_x_idxs[0]) and (i < wsi_x_idxs[1]):
            alpha = 1
            linewidth = 3
        else:
            alpha = 0.7
            linewidth = 1.5

        plt.scatter(
            x_embedded[i, 0],
            x_embedded[i, 1],
            c=c,
            marker=marker,
            label=label,
            alpha=alpha,
            linewidth=linewidth,
        )

    elapsed_time = time.time() - start_time
    logging.info(f"elapsed_time({method}): {elapsed_time:.1f} [sec]")

    plt.title(f"{title} with {method}")
    if output_dir is not None:
        plt.savefig(f"{output_dir}{title}_{method}.png", format="png", dpi=300)
    # plt.show()
    plt.clf()
    plt.close()


# plot時にrandomにサンプリング
def plot_wsi_distrib_in_all_cl_sampled(
    x_embedded,
    sampled_x_idxs: list,
    wsi_x_idxs: list,
    cluster_labels: list = None,
    cl_labels: list = None,
    method: str = "tsne",
    output_dir: str = None,
    title: str = "feature_space",
    markers: list = markers,
):
    print(f"Plot feature space with {method}")
    plt.figure(figsize=(9, 6))
    cmap = plt.get_cmap("tab10")

    start_time = time.time()

    print("Plot all samples")
    for x_idx in tqdm(sampled_x_idxs):
        cluster_label = cluster_labels[x_idx]
        cl_label = cl_labels[x_idx]

        c = [cmap(cluster_label)]
        marker = markers[cl_label]

        plt.scatter(
            x_embedded[x_idx, 0],
            x_embedded[x_idx, 1],
            c=c,
            marker=marker,
            label=None,
            alpha=0.5,
            linewidth=0,
        )

    print("Plot taget-wsi samples")
    for x_idx in tqdm(wsi_x_idxs):
        cluster_label = cluster_labels[x_idx]
        cl_label = cl_labels[x_idx]

        c = [cmap(cluster_label)]
        marker = markers[cl_label]

        plt.scatter(
            x_embedded[x_idx, 0],
            x_embedded[x_idx, 1],
            c=c,
            marker=marker,
            label=None,
            alpha=1,
            linewidth=1.0,
            edgecolors="black",
        )

    # 凡例用
    plt.scatter([], [], c="black", alpha=1, marker=markers[0], label="Non-Neop")
    plt.scatter([], [], c="black", alpha=1, marker=markers[2], label="LSIL")
    plt.scatter([], [], c="black", alpha=1, marker=markers[1], label="HSIL")

    elapsed_time = time.time() - start_time
    logging.info(f"elapsed_time: {elapsed_time:.1f} [sec]")

    plt.title(f"{title} with {method}")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    if output_dir is not None:
        plt.savefig(
            f"{output_dir}{title}_{method}.png",
            format="png",
            dpi=300,
            bbox_inches="tight",
        )
    # plt.show()
    plt.clf()
    plt.close()


def plot_img_feature_space(x, labels, image_list, method="tsne", zoom=0.1):
    print(f"Visualize features with {method}")
    # fig, ax = plt.subplots(figsize=(30, 30))
    fig, ax = plt.subplots(figsize=(100, 100))

    x_embedded = get_embed_feature(x, method)

    imscatter(x_embedded[:, 0], x_embedded[:, 1], labels, image_list, ax=ax, zoom=zoom)

    plt.title(f"Embedding Space with {method}")
    plt.show()
    plt.clf()
    plt.close()


def plot_feature_space_domains(
    x_lists: list,
    method: str = "tsne",
    output_dir: str = None,
    title: str = "feature_space",
    domain_labels: list = None,
    cl_labels: list = None,
    colors_domains: list = colors_domains,
    markers: list = markers,
):
    print(f"Visualize features with {method}")
    plt.figure(figsize=(8, 6))

    start_time = time.time()

    domain_idx_list = []
    cl_idx_list = []
    num_x_list = []
    init_flag = True

    for domain_idx, x_list in enumerate(x_lists):
        if len(x_list) == 0:
            continue

        for cl_idx, x in enumerate(x_list):
            domain_idx_list.append(domain_idx)
            cl_idx_list.append(cl_idx)
            num_x_list.append(x.shape[0])
            if init_flag:
                x_all = x
                init_flag = False
            else:
                x_all = np.concatenate([x_all, x], axis=0)

    x_embedded = get_embed_feature(x_all, method)

    for i, (domain_idx, cl_idx, num_x) in enumerate(
        zip(domain_idx_list, cl_idx_list, num_x_list)
    ):
        c = [edgecolor(label=cl_idx, cols=colors_domains[domain_idx])]
        if (domain_labels is not None) and (cl_labels is not None):
            label = f"{domain_labels[domain_idx]}_{cl_labels[cl_idx]}"
        else:
            label = f"dataset{domain_idx}_cl{cl_idx}"
        plt.scatter(
            x_embedded[i * num_x: (i * num_x) + num_x, 0],
            x_embedded[i * num_x: (i * num_x) + num_x, 1],
            c=c,
            marker=markers[domain_idx],
            label=label,
        )

    elapsed_time = time.time() - start_time
    logging.info(f"elapsed_time({method}): {elapsed_time:.1f} [sec]")

    # plt.colorbar()
    plt.title(f"Embedding Space with {method}")
    plt.legend()
    if output_dir is not None:
        plt.savefig(f"{output_dir}{title}_{method}.png", format="png", dpi=300)
    # plt.show()
    plt.clf()
    plt.close()
