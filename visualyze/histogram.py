import matplotlib.pyplot as plt
import numpy as np


def plot_cluster_hist(
    wsi_clusters_nd,
    cluster_total_num: int,
    title: str,
    output_dir: str = "./",
    subtitle: str = "",
):
    """
    あるwsiのパッチが所属するクラスターのヒストグラムを作成
    Parameters
        ----------
        wsi_clusters_nd : numpy.ndarray
            対象wsiの各パッチが所属するクラスターのラベル情報が入ったndarray
        cluster_total_num : int
            クラスターの総数

        Returns
        -------
        None
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    max_cluster_id = cluster_total_num - 1

    ax1.hist(
        wsi_clusters_nd,
        range=(-0.5, max_cluster_id + 0.5),
        density=True,
        alpha=1,
        rwidth=0.8,
    )
    if len(subtitle) > 0:
        fig_title = f"{title}\n{subtitle}"
    else:
        fig_title = title
    ax1.set_title(fig_title)
    ax1.set_xlabel("Cluster ID")
    ax1.set_ylabel("Freq")
    ax1.set_xlim(-0.5, max_cluster_id + 0.5)
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(np.arange(0, cluster_total_num, step=1))
    fig.tight_layout()
    fig.savefig(f"{output_dir}{title}.png", dpi=300)
    plt.close()


def plot_cluster_wsinum_hist(
    wsi_clusters_nd,
    cluster_total_num: int,
    title: str,
    output_dir: str = "./",
    subtitle: str = "",
):
    """
    あるwsiのパッチが所属するクラスターのヒストグラムを作成
    Parameters
        ----------
        wsi_clusters_nd : numpy.ndarray
            対象wsiの各パッチが所属するクラスターのラベル情報が入ったndarray
        cluster_total_num : int
            クラスターの総数

        Returns
        -------
        None
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    max_cluster_id = cluster_total_num - 1

    ax1.hist(
        wsi_clusters_nd,
        range=(-0.5, max_cluster_id + 0.5),
        density=True,
        alpha=1,
        rwidth=0.8,
    )
    if len(subtitle) > 0:
        fig_title = f"{title}\n{subtitle}"
    else:
        fig_title = title
    ax1.set_title(fig_title)
    ax1.set_xlabel("Cluster ID")
    ax1.set_ylabel("Freq")
    ax1.set_xlim(-0.5, max_cluster_id + 0.5)
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(np.arange(0, cluster_total_num, step=1))
    fig.tight_layout()
    fig.savefig(f"{output_dir}{title}.png", dpi=300)
    plt.close()
