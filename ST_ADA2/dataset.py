import os
import sys
import re
import random
import glob
from natsort import natsorted
import numpy as np
import torch
from torchvision import transforms
from PIL import Image


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from S.dataset import WSI, WSIDataset
from clustering.cluster_sampler import get_cluster_ids


class WSI_cluster(torch.utils.data.Dataset):
    """
    file_listの全パッチの特徴量でクラスタリング
        → 各パッチにクラスターラベルを付与
    """

    def __init__(
        self, file_list: list, cluster_id_list: list, shape: tuple = None, transform=None
    ):
        self.file_list = file_list
        self.cluster_id_list = cluster_id_list
        self.shape = shape
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def preprocess(self, img_pil):
        if self.transform is not None:
            if self.transform["Resize"]:
                img_pil = transforms.Resize(self.shape, interpolation=0)(img_pil)
            if self.transform["HFlip"]:
                img_pil = transforms.RandomHorizontalFlip(0.5)(img_pil)
            if self.transform["VFlip"]:
                img_pil = transforms.RandomVerticalFlip(0.5)(img_pil)
        return np.asarray(img_pil)

    def transpose(self, img):
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)

        # HWC to CHW
        img_trans = img.transpose((2, 0, 1))
        # For rgb or grayscale image
        if img_trans.max() > 1:
            img_trans = img_trans / 255
        return img_trans

    def __getitem__(self, i):
        img_file = self.file_list[i]
        cluster_id = np.array(self.cluster_id_list[i])
        img_pil = Image.open(img_file)
        if img_pil.mode != "RGB":
            img_pil = img_pil.convert("RGB")

        img = self.preprocess(img_pil)
        img = self.transpose(img)

        item = {
            "image": torch.from_numpy(img).type(torch.FloatTensor),
            "cluster_id": torch.from_numpy(cluster_id).type(torch.long),
            "name": img_file,
        }

        return item


class WSIDataset_ST1_ADA2_ValT(WSIDataset):
    """
    Adversarial Domain Adaptation用
    labeled sourceのWSIとsingle labeled target WSI, unlabeled target WSIを使用
    valid dataには複数枚のtarget WSIを使用
    classification用のdataloaderとdiscriminator用のdataloaderは別々に用意
        - classification用: class-balanced
        - disriminator用: cluster-balanced
    """

    def __init__(
        self,
        l_src_wsis: list,
        l_trg_wsi: str,
        unl_trg_wsis: list,
        trg_valid_wsis: list,
        src_imgs_dir: str,
        trg_imgs_dir: str,
        classes: list = [0, 1, 2],
        shape: tuple = (256, 256),
        transform: dict = None,
        batch_size: int = 16,
        cluster_num: int = 10,
        weight_path: str = None,
        balance_domain: bool = False,
    ):
        self.l_src_wsis = l_src_wsis
        self.l_trg_wsi = l_trg_wsi
        self.unl_trg_wsis = unl_trg_wsis
        self.trg_valid_wsis = trg_valid_wsis
        self.src_imgs_dir = src_imgs_dir
        self.trg_imgs_dir = trg_imgs_dir
        self.classes = classes
        self.shape = shape
        self.transform = transform
        self.sub_classes = self.get_sub_classes()

        # labeled sourceのWSIリストから，全patchのpathリストを取得
        self.l_src_files = self.get_files(self.l_src_wsis, self.src_imgs_dir)
        # labeled targetのWSIリストから，全patchのpathリストを取得
        self.l_trg_files = self.get_files([self.l_trg_wsi], self.trg_imgs_dir)
        # unlabeled targetのWSIリストから，全patchのpathリストを取得
        self.unl_trg_files = self.get_files(self.unl_trg_wsis, self.trg_imgs_dir)

        # validation用のtargetのWSIリストから，全patchのpathリストを取得
        self.trg_valid_files = self.get_files(self.trg_valid_wsis, self.trg_imgs_dir)

        # # # FIXME: Debug用 ------------ #
        # self.l_src_files = self.l_src_files[:512]
        # # self.l_trg_files = self.l_trg_files[:512]
        # self.unl_trg_files = self.unl_trg_files[:512]
        # # # -------------------------- #

        # l_src_filesとl_trg_filesを同数にする
        if balance_domain:
            self.rebalance()

        train_files = self.l_src_files + self.l_trg_files + self.unl_trg_files
        valid_files = self.trg_valid_files
        test_files = self.unl_trg_files  # unl_trgがtest対象

        # self.data_len = len(train_files) + len(valid_files)
        print(f"[wsi (source)]  train: (l) {len(self.l_src_wsis)}")
        print(
            f"[wsi (target)]  train: (l) 1, (unl) {len(self.unl_trg_wsis)}, valid: {len(self.trg_valid_wsis)}"
        )
        print(
            f"[wsi (all)]  train: {len(self.l_src_wsis) + len(self.unl_trg_wsis) + 1}, valid: {len(self.trg_valid_wsis)}"
        )
        print(f"[patch (source)] train: {len(self.l_src_files)}")
        print(
            f"[patch (target)] train: (l) {len(self.l_trg_files)}, (unl) {len(self.unl_trg_files)}, valid: {len(self.trg_valid_files)}"
        )
        print(
            f"[patch (all)] train: {len(train_files)}, valid: {len(valid_files)}"
        )
        print("test data: unl_trg")

        # classification用 (classラベルあり)
        self.clf_l_src_data = WSI(self.l_src_files, self.classes, self.shape, self.transform)
        self.clf_l_trg_data = WSI(self.l_trg_files, self.classes, self.shape, self.transform)

        print("get src_cluster_ids...")
        self.src_cluster_ids = get_cluster_ids(
            file_list=self.l_src_files,
            weight_path=weight_path,
            cluster_num=cluster_num,
            batch_size=batch_size,
            shape=shape,
            classes=classes,
        )

        print("get trg_cluster_ids...")
        self.trg_cluster_ids = get_cluster_ids(
            file_list=self.unl_trg_files,
            weight_path=weight_path,
            cluster_num=cluster_num,
            batch_size=batch_size,
            shape=shape,
            classes=classes,
        )

        self.d_src_data = WSI_cluster(self.l_src_files, self.src_cluster_ids, self.shape, self.transform)
        self.d_trg_data = WSI_cluster(self.unl_trg_files, self.trg_cluster_ids, self.shape, self.transform)

        # classificationのvalidation/test用 (classラベル有り)
        test_transform = self.transform.copy()
        test_transform["HFlip"] = False
        test_transform["VFlip"] = False
        self.valid_data = WSI(valid_files, self.classes, self.shape, test_transform)
        self.test_data = WSI(test_files, self.classes, self.shape, test_transform)

    def get_files(self, wsis: list, imgs_dir: str):
        re_pattern = re.compile("|".join([f"/{i}/" for i in self.sub_classes]))

        files_list = []
        for wsi in wsis:
            files_list.extend(
                [
                    p
                    for p in glob.glob(imgs_dir + f"*/{wsi}_*/*.png", recursive=True)
                    if bool(re_pattern.search(p))
                ]
            )
        return files_list

    def get_wsi_split(self):
        return (
            natsorted(self.l_src_wsis + [self.l_trg_wsi] + self.unl_trg_wsis),
            natsorted(self.trg_valid_wsis),
        )

    def wsi_num(self):
        wsi_num = len(self.l_src_wsis + [self.l_trg_wsi] + self.unl_trg_wsis)
        wsi_num += len(self.trg_valid_wsis)
        return len(wsi_num)

    def rebalance(self):
        """
        labeled sourceとlabeled targetでパッチ数を同数になるよう増量
        """
        random.seed(0)
        l_src_num = len(self.l_src_files)
        l_trg_num = len(self.l_trg_files)
        if l_src_num > l_trg_num:  # targetの方が少ない場合
            add_num = l_src_num - l_trg_num
            self.l_trg_files += \
                random.choices(self.l_trg_files, k=add_num)
        else:  # sourceの方が少ない場合
            add_num = l_trg_num - l_src_num
            self.l_src_files += \
                random.choices(self.l_src_files, k=add_num)

        assert len(self.l_src_files) == len(self.l_trg_files), \
            f"l_src_files: {len(self.l_src_files)}, l_trg_files: {len(self.l_trg_files)}"

    def get(self):
        return self.clf_l_src_data, self.clf_l_trg_data, self.d_src_data, self.d_trg_data, self.valid_data, self.test_data
