import os
import sys
import torch
import numpy as np
import logging
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.decomposition import PCA
from umap import UMAP

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from S.dataset import WSI
from visualyze2.vis_utils import get_files_oneclass


def get_embed_feature(x, method: str):
    def tsne_embed(x):
        tsne = TSNE(n_components=2, random_state=0, perplexity=30)
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
    wsi_list: list,
    imgs_dir: str,
    sample_N: int = 500,
    classes: list = [0, 1, 2],
    input_shape: tuple = (256, 256),
    batch_size: int = 32,
    class_idx: int = 0,
):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 0, "pin_memory": True} if use_cuda else {}

    file_list = get_files_oneclass(
        wsi_list=wsi_list,
        classes=classes,
        imgs_dir=imgs_dir,
        N=sample_N,
        cl_idx=class_idx,
    )

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
    wsi_list: list,
    imgs_dir: str,
    sample_N: int = 500,
    classes: list = [0, 1, 2],
    input_shape: tuple = (256, 256),
    batch_size: int = 32,
    output_dir: str = None,
    title: str = "latent_vecs"
):
    latent_vecs_list = []
    for cl_idx in range(len(classes)):
        # クラスごとに特徴量ベクトルを取得
        latent_vecs = get_latent_vecs(
            model, wsi_list=wsi_list,
            imgs_dir=imgs_dir,
            sample_N=sample_N,
            classes=classes,
            input_shape=input_shape,
            batch_size=batch_size,
            class_idx=cl_idx
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


# class WSI(torch.utils.data.Dataset):
#     def __init__(self, file_list, classes=[0, 1, 2, 3], shape=None, transform=None, is_pred=False):
#         self.file_list = file_list
#         self.classes = classes
#         self.shape = shape
#         self.transform = transform
#         self.is_pred = is_pred

#     def __len__(self):
#         return len(self.file_list)

#     # pathからlabelを取得
#     def get_label(self, path):
#         def check_path(cl, path):
#             if f"/{cl}/" in path:
#                 return True
#             else:
#                 return False

#         for idx in range(len(self.classes)):
#             cl = self.classes[idx]

#             if isinstance(cl, list):
#                 for sub_cl in cl:
#                     if check_path(sub_cl, path):
#                         label = idx
#             else:
#                 if check_path(cl, path):
#                     label = idx
#         assert label is not None, "label is not included in {path}"
#         return np.array(label)

#     def preprocess(self, img_pil):
#         if self.transform is not None:
#             if self.transform['Resize']:
#                 img_pil = transforms.Resize(
#                     self.shape, interpolation=0
#                 )(img_pil)
#             if self.transform['HFlip']:
#                 img_pil = transforms.RandomHorizontalFlip(0.5)(img_pil)
#             if self.transform['VFlip']:
#                 img_pil = transforms.RandomVerticalFlip(0.5)(img_pil)
#         return np.asarray(img_pil)

#     def transpose(self, img):
#         if len(img.shape) == 2:
#             img = np.expand_dims(img, axis=2)

#         # HWC to CHW
#         img_trans = img.transpose((2, 0, 1))
#         # For rgb or grayscale image
#         if img_trans.max() > 1:
#             img_trans = img_trans / 255
#         return img_trans

#     def __getitem__(self, i):
#         img_file = self.file_list[i]
#         img_pil = Image.open(img_file)
#         if img_pil.mode != 'RGB':
#             img_pil = img_pil.convert('RGB')

#         img = self.preprocess(img_pil)
#         img = self.transpose(img)

#         if self.is_pred:
#             item = {
#                 'image': torch.from_numpy(img).type(torch.FloatTensor),
#                 'name': img_file
#             }
#         else:
#             label = self.get_label(img_file)
#             item = {
#                 'image': torch.from_numpy(img).type(torch.FloatTensor),
#                 'label': torch.from_numpy(label).type(torch.long),
#                 'name': img_file
#             }

#         return item
