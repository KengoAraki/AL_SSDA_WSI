import os
import sys
import torch
import numpy as np
import logging
from tqdm import tqdm
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.decomposition import PCA
from umap import UMAP
import random

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from S.dataset import WSI
from plot_miccai.vis_utils import get_files_oneclass


def get_embed_feature(x, method: str):
    def tsne_embed(x):
        tsne = TSNE(n_components=2, random_state=0, perplexity=30, n_jobs=8)
        # tsne = TSNE(n_components=2, random_state=0, perplexity=5)
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


def extract_feature(model, device, test_loader, n_data):
    model.eval()
    init_flag = True
    with torch.no_grad():
        with tqdm(total=n_data, unit="img") as pbar:
            for batch in test_loader:
                data, target = batch["image"], batch["label"]
                data, target = data.to(device), target.to(device)
                latent_vecs = model(data)

                latent_vecs = latent_vecs.cpu()
                target = target.cpu()
                if init_flag:
                    latent_vecs_stack = latent_vecs
                    target_stack = target
                    init_flag = False
                else:
                    latent_vecs_stack = torch.cat((latent_vecs_stack, latent_vecs), 0)
                    target_stack = torch.cat((target_stack, target), 0)
                pbar.update(target.shape[0])
    return latent_vecs_stack, target_stack


def get_latent_vecs(
    model,
    file_list: list,
    classes: list = [0, 1, 2],
    input_shape: tuple = (256, 256),
    batch_size: int = 32,
):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 0, "pin_memory": True} if use_cuda else {}

    if len(file_list) > 0:
        transform = {"Resize": True, "HFlip": False, "VFlip": False}
        dataset = WSI(
            file_list, classes, input_shape, transform, is_pred=False
        )
        dataset_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, **kwargs
        )

        latent_vecs, _ = extract_feature(
            model, device, dataset_loader, len(dataset)
        )
        latent_vecs = latent_vecs.numpy()
    else:  # このクラスのデータが存在しない場合
        latent_vecs = np.array([])
    return latent_vecs


def get_latent_vecs_list(
    model,
    all_class_files_list: list,  # 全クラスのファイルリストのリスト
    classes: list = [0, 1, 2],
    input_shape: tuple = (256, 256),
    batch_size: int = 32,
    output_dir: str = None,
    title: str = "latent_vecs"
):
    # all_class_files_list: [クラス0のfile list, クラス1のfile list, ..., クラスnのfile list]

    latent_vecs_list = []
    for cl_idx in range(len(classes)):
        files = all_class_files_list[cl_idx]
        # クラスごとに特徴量ベクトルを取得
        latent_vecs = get_latent_vecs(
            model,
            file_list=files,
            input_shape=input_shape,
            batch_size=batch_size,
        )
        # 特徴ベクトルを保存
        if output_dir is not None:
            np.save(
                f"{output_dir}{title}_cl{cl_idx}",
                latent_vecs,
            )

        latent_vecs_list.append(latent_vecs)
    logging.info(f"latent_vecs: {latent_vecs.shape}")
    return latent_vecs_list


# 各クラスのfile listのlistを取得
def get_all_class_files_list(
    wsi_list: list,
    imgs_dir: str,
    classes: list = [0, 1, 2],
):
    random.seed(0)
    all_class_files_list = []
    for class_idx in range(len(classes)):
        file_list = get_files_oneclass(
            wsi_list=wsi_list,
            classes=classes,
            imgs_dir=imgs_dir,
            cl_idx=class_idx,
        )
        random.shuffle(file_list)
        all_class_files_list.append(file_list)
    return all_class_files_list
