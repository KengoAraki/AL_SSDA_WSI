import os
import glob
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from natsort import natsorted
import re
import copy


def get_files(wsis, classes, imgs_dir):
    def get_sub_classes(classes):
        # classesからsub-classを取得
        sub_cl_list = []
        for idx in range(len(classes)):
            cl = classes[idx]
            if isinstance(cl, list):
                for sub_cl in cl:
                    sub_cl_list.append(sub_cl)
            else:
                sub_cl_list.append(cl)
        return sub_cl_list

    re_pattern = re.compile("|".join([f"/{i}/" for i in get_sub_classes(classes)]))

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


class WSI(torch.utils.data.Dataset):
    def __init__(
        self, file_list, classes=[0, 1, 2, 3], shape=None, transform=None, is_pred=False
    ):
        self.file_list = file_list
        self.classes = classes
        self.shape = shape
        self.transform = transform
        self.is_pred = is_pred

    def __len__(self):
        return len(self.file_list)

    # pathからlabelを取得
    def get_label(self, path):
        def check_path(cl, path):
            if f"/{cl}/" in path:
                return True
            else:
                return False

        for idx in range(len(self.classes)):
            cl = self.classes[idx]

            if isinstance(cl, list):
                for sub_cl in cl:
                    if check_path(sub_cl, path):
                        label = idx
            else:
                if check_path(cl, path):
                    label = idx
        assert label is not None, "label is not included in {path}"
        return np.array(label)

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
        img_pil = Image.open(img_file)
        if img_pil.mode != "RGB":
            img_pil = img_pil.convert("RGB")

        img = self.preprocess(img_pil)
        img = self.transpose(img)

        if self.is_pred:
            item = {
                "image": torch.from_numpy(img).type(torch.FloatTensor),
                "name": img_file,
            }
        else:
            label = self.get_label(img_file)
            item = {
                "image": torch.from_numpy(img).type(torch.FloatTensor),
                "label": torch.from_numpy(label).type(torch.long),
                "name": img_file,
            }

        return item


class WSIDataset(object):
    def __init__(
        self,
        train_wsis,
        valid_wsis,
        test_wsis,
        imgs_dir: str,
        classes: list = [0, 1, 2, 3],
        shape: tuple = (512, 512),
        transform: dict = None,
    ):
        self.train_wsis = train_wsis
        self.valid_wsis = valid_wsis
        self.test_wsis = test_wsis

        self.imgs_dir = imgs_dir
        self.classes = classes
        self.shape = shape
        self.transform = transform
        self.sub_classes = self.get_sub_classes()

        self.wsi_list = []
        for i in range(len(self.sub_classes)):
            sub_cl = self.sub_classes[i]
            self.wsi_list.extend(
                [p[:-4] for p in os.listdir(self.imgs_dir + f"{sub_cl}/")]
            )
        self.wsi_list = list(set(self.wsi_list))
        # os.listdirによる実行時における要素の順不同対策のため
        self.wsi_list = natsorted(self.wsi_list)

        train_files = self.get_files(self.train_wsis)
        valid_files = self.get_files(self.valid_wsis)
        test_files = self.get_files(self.test_wsis)

        self.data_len = len(train_files) + len(valid_files) + len(test_files)
        print(
            f"[wsi]  train: {len(self.train_wsis)}, valid: {len(self.valid_wsis)}, test: {len(self.test_wsis)}"
        )
        print(
            f"[patch] train: {len(train_files)}, valid: {len(valid_files)}, test: {len(test_files)}"
        )

        test_files = natsorted(test_files)

        self.train_data = WSI(train_files, self.classes, self.shape, self.transform)

        test_transform = self.transform.copy()
        test_transform["HFlip"] = False
        test_transform["VFlip"] = False
        self.valid_data = WSI(valid_files, self.classes, self.shape, test_transform)
        self.test_data = WSI(test_files, self.classes, self.shape, test_transform)

    def __len__(self):
        # return len(self.file_list)
        return len(self.data_len)

    def get_sub_classes(self):
        # classesからsub-classを取得
        sub_cl_list = []
        for idx in range(len(self.classes)):
            cl = self.classes[idx]
            if isinstance(cl, list):
                for sub_cl in cl:
                    sub_cl_list.append(sub_cl)
            else:
                sub_cl_list.append(cl)
        return sub_cl_list

    def get_files(self, wsis):
        re_pattern = re.compile("|".join([f"/{i}/" for i in self.sub_classes]))

        files_list = []
        for wsi in wsis:
            files_list.extend(
                [
                    p
                    for p in glob.glob(
                        self.imgs_dir + f"*/{wsi}_*/*.png", recursive=True
                    )
                    if bool(re_pattern.search(p))
                ]
            )
        return files_list

    def get_wsi_split(self):
        return (
            natsorted(self.train_wsis),
            natsorted(self.valid_wsis),
            natsorted(self.test_wsis),
        )

    def get_wsi_num(self):
        return len(self.wsi_list)

    def get(self):
        return self.train_data, self.valid_data, self.test_data


class WSIDatasetST1(WSIDataset):
    """
    source dataset + single target WSI 用
    """

    def __init__(
        self,
        src_train_wsis: list,
        src_valid_wsis: list,
        src_test_wsis: list,
        trg_all_wsis: list,
        trg_selected_wsi: str,
        src_imgs_dir: str,
        trg_imgs_dir: str,
        classes: list = [0, 1, 2, 3],
        shape: tuple = (512, 512),
        transform: dict = None,
    ):
        self.src_train_wsis = src_train_wsis
        self.src_valid_wsis = src_valid_wsis
        self.src_test_wsis = src_test_wsis
        self.trg_all_wsis = trg_all_wsis
        self.trg_selected_wsi = trg_selected_wsi
        self.src_imgs_dir = src_imgs_dir
        self.trg_imgs_dir = trg_imgs_dir
        self.classes = classes
        self.shape = shape
        self.transform = transform
        self.sub_classes = self.get_sub_classes()

        # 分類対象クラスを有するWSIのリストを取得
        self.src_wsi_list = self.get_wsi_list(self.src_imgs_dir)
        self.trg_wsi_list = self.get_wsi_list(self.trg_imgs_dir)
        self.wsi_list = self.src_wsi_list + self.trg_wsi_list

        # 各WSIのリストからパッチのパスリストを取得 (source)
        src_train_files = self.get_files(self.src_train_wsis, self.src_imgs_dir)
        src_valid_files = self.get_files(self.src_valid_wsis, self.src_imgs_dir)
        src_test_files = self.get_files(self.src_test_wsis, self.src_imgs_dir)

        # 各WSIのリストからパッチのパスリストを取得 (target)
        trg_train_files = self.get_files([self.trg_selected_wsi], self.trg_imgs_dir)
        self.trg_test_wsis = copy.deepcopy(trg_all_wsis)
        self.trg_test_wsis.remove(self.trg_selected_wsi)
        trg_test_files = self.get_files(self.trg_test_wsis, self.trg_imgs_dir)

        train_files = src_train_files + trg_train_files
        valid_files = src_valid_files
        test_files = src_test_files + trg_test_files
        test_files = natsorted(test_files)

        self.data_len = len(train_files) + len(valid_files) + len(test_files)
        print(
            f"[wsi (source)]  train: {len(self.src_train_wsis)}, valid: {len(self.src_valid_wsis)}, test: {len(self.src_test_wsis)}"
        )
        print(
            f"[patch (source)] train: {len(src_train_files)}, valid: {len(src_valid_files)}, test: {len(src_test_files)}"
        )
        print(f"[wsi (target)]  train: 1, test: {len(self.trg_test_wsis)}")
        print(
            f"[patch (target)] train: {len(trg_train_files)}, test: {len(trg_test_files)}"
        )
        print(
            f"[wsi]  train: {len(self.src_train_wsis)+1}, valid: {len(self.src_valid_wsis)}, test: {len(self.src_test_wsis + self.trg_test_wsis)}"
        )
        print(
            f"[patch] train: {len(train_files)}, valid: {len(valid_files)}, test: {len(test_files)}"
        )

        self.train_data = WSI(train_files, self.classes, self.shape, self.transform)

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

    def get_wsi_list(self, imgs_dir: str):
        wsi_list = []
        for i in range(len(self.sub_classes)):
            sub_cl = self.sub_classes[i]
            wsi_list.extend([p[:-4] for p in os.listdir(imgs_dir + f"{sub_cl}/")])
        wsi_list = list(set(wsi_list))
        # os.listdirによる実行時における要素の順不同対策のため
        return natsorted(wsi_list)

    def get_wsi_split(self):
        return (
            natsorted(self.src_train_wsis + [self.trg_selected_wsi]),
            natsorted(self.src_valid_wsis),
            natsorted(self.src_test_wsis + self.trg_test_wsis),
        )


class WSIDatasetT1(WSIDataset):
    """
    single target WSIを用いたFine-tuning 用
    """

    def __init__(
        self,
        trg_all_wsis: list,
        trg_selected_wsi: str,
        trg_imgs_dir: str,
        classes: list = [0, 1, 2, 3],
        shape: tuple = (256, 256),
        transform: dict = None,
        val_ratio: float = 0.2,
    ):
        self.trg_all_wsis = trg_all_wsis
        self.trg_selected_wsi = trg_selected_wsi
        self.trg_imgs_dir = trg_imgs_dir
        self.classes = classes
        self.shape = shape
        self.transform = transform
        self.val_ratio = val_ratio
        self.sub_classes = self.get_sub_classes()

        # 分類対象クラスを有するWSIのリストを取得
        self.wsi_list = self.get_wsi_list(self.trg_imgs_dir)

        # 各WSIのリストからパッチのパスリストを取得 (target)
        trg_trvl_files = self.get_files([self.trg_selected_wsi], self.trg_imgs_dir)
        self.trg_test_wsis = copy.deepcopy(trg_all_wsis)
        self.trg_test_wsis.remove(self.trg_selected_wsi)
        trg_test_files = self.get_files(self.trg_test_wsis, self.trg_imgs_dir)

        # train/valid用のWSIのパッチをtrain, validにランダムに割当
        trg_train_files, trg_valid_files = self.split_trvl_files(trg_trvl_files)

        train_files = trg_train_files
        valid_files = trg_valid_files
        test_files = natsorted(trg_test_files)

        self.data_len = len(train_files) + len(valid_files) + len(test_files)
        print(
            f"[wsi (target)]  train: 1, valid: 1 (same WSI as train), test: {len(self.trg_test_wsis)}"
        )
        print(
            f"[patch (target)] train: {len(train_files)}, valid: {len(valid_files)}, test: {len(test_files)}"
        )

        self.train_data = WSI(train_files, self.classes, self.shape, self.transform)

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

    def get_wsi_list(self, imgs_dir: str):
        wsi_list = []
        for i in range(len(self.sub_classes)):
            sub_cl = self.sub_classes[i]
            wsi_list.extend([p[:-4] for p in os.listdir(imgs_dir + f"{sub_cl}/")])
        wsi_list = list(set(wsi_list))
        # os.listdirによる実行時における要素の順不同対策のため
        return natsorted(wsi_list)

    def get_wsi_split(self):
        return (
            [self.trg_selected_wsi],
            [self.trg_selected_wsi],
            natsorted(self.trg_test_wsis),
        )

    def split_trvl_files(self, trvl_files: list):
        # train/valid用のWSIのパッチをtrain, valid用に分割
        from sklearn.model_selection import train_test_split

        train_files, valid_files = train_test_split(
            trvl_files, test_size=self.val_ratio, random_state=0
        )
        return train_files, valid_files


class WSIDatasetT1_ValT(WSIDataset):
    """
    single target WSIを用いたFine-tuning 用
    valid dataには複数枚のtarget WSIを使用
    """

    def __init__(
        self,
        trg_all_wsis: list,
        trg_selected_wsi: str,
        trg_valid_wsis: list,
        trg_imgs_dir: str,
        classes: list = [0, 1, 2, 3],
        shape: tuple = (256, 256),
        transform: dict = None,
    ):
        self.trg_all_wsis = trg_all_wsis
        self.trg_selected_wsi = trg_selected_wsi
        self.trg_valid_wsis = trg_valid_wsis
        self.trg_imgs_dir = trg_imgs_dir
        self.classes = classes
        self.shape = shape
        self.transform = transform
        self.sub_classes = self.get_sub_classes()

        # 分類対象クラスを有するWSIのリストを取得
        self.wsi_list = self.get_wsi_list(self.trg_imgs_dir)

        self.trg_test_wsis = copy.deepcopy(trg_all_wsis)
        # train用のWSIをターゲット全体のリストから取り除く
        self.trg_test_wsis.remove(self.trg_selected_wsi)
        # valid用のWSIをターゲット全体のリストから取り除く
        for valid_wsi in self.trg_valid_wsis:
            self.trg_test_wsis.remove(valid_wsi)

        # 各WSIのリストからパッチのパスリストを取得 (target)
        trg_train_files = self.get_files([self.trg_selected_wsi], self.trg_imgs_dir)
        trg_valid_files = self.get_files(self.trg_valid_wsis, self.trg_imgs_dir)
        trg_test_files = self.get_files(self.trg_test_wsis, self.trg_imgs_dir)

        train_files = trg_train_files
        valid_files = trg_valid_files
        test_files = natsorted(trg_test_files)

        self.data_len = len(train_files) + len(valid_files) + len(test_files)
        print(
            f"[wsi (target)]  train: 1, valid: {len(self.trg_valid_wsis)}, test: {len(self.trg_test_wsis)}"
        )
        print(
            f"[patch (target)] train: {len(train_files)}, valid: {len(valid_files)}, test: {len(test_files)}"
        )

        self.train_data = WSI(train_files, self.classes, self.shape, self.transform)

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

    def get_wsi_list(self, imgs_dir: str):
        wsi_list = []
        for i in range(len(self.sub_classes)):
            sub_cl = self.sub_classes[i]
            wsi_list.extend([p[:-4] for p in os.listdir(imgs_dir + f"{sub_cl}/")])
        wsi_list = list(set(wsi_list))
        # os.listdirによる実行時における要素の順不同対策のため
        return natsorted(wsi_list)

    def get_wsi_split(self):
        return (
            [self.trg_selected_wsi],
            [self.trg_selected_wsi],
            natsorted(self.trg_test_wsis),
        )


class WSIDatasetST1_ValT(WSIDataset):
    """
    sourceのWSIとsingle target WSIを用いたFine-tuning 用
    valid dataには複数枚のtarget WSIを使用
    """

    def __init__(
        self,
        src_train_wsis: list,
        trg_all_wsis: list,
        trg_selected_wsi: str,
        trg_valid_wsis: list,
        src_imgs_dir: str,
        trg_imgs_dir: str,
        classes: list = [0, 1, 2, 3],
        shape: tuple = (256, 256),
        transform: dict = None,
    ):
        self.src_train_wsis = src_train_wsis
        self.trg_all_wsis = trg_all_wsis
        self.trg_selected_wsi = trg_selected_wsi
        self.trg_valid_wsis = trg_valid_wsis
        self.src_imgs_dir = src_imgs_dir
        self.trg_imgs_dir = trg_imgs_dir
        self.classes = classes
        self.shape = shape
        self.transform = transform
        self.sub_classes = self.get_sub_classes()

        # 分類対象クラスを有するWSIのリストを取得
        self.wsi_list = self.get_wsi_list(self.trg_imgs_dir)

        # 各WSIのリストからパッチのパスリストを取得 (target)
        self.trg_test_wsis = copy.deepcopy(trg_all_wsis)
        # train用のWSIをターゲット全体のリストから取り除く
        self.trg_test_wsis.remove(self.trg_selected_wsi)
        # valid用のWSIをターゲット全体のリストから取り除く
        for valid_wsi in self.trg_valid_wsis:
            self.trg_test_wsis.remove(valid_wsi)

        # tarin用のWSIのリストからパッチのパスリストを取得 (source)
        src_train_files = self.get_files(self.src_train_wsis, self.src_imgs_dir)

        # 各WSIのリストからパッチのパスリストを取得 (target)
        trg_train_files = self.get_files([self.trg_selected_wsi], self.trg_imgs_dir)
        trg_valid_files = self.get_files(self.trg_valid_wsis, self.trg_imgs_dir)
        trg_test_files = self.get_files(self.trg_test_wsis, self.trg_imgs_dir)

        train_files = src_train_files + trg_train_files
        valid_files = trg_valid_files
        test_files = natsorted(trg_test_files)

        self.data_len = len(train_files) + len(valid_files) + len(test_files)
        print(f"[wsi (source)]  train: {len(self.src_train_wsis)}")
        print(f"[patch (source)] train: {len(src_train_files)}")
        print(
            f"[wsi (target)]  train: 1, valid: {len(self.trg_valid_wsis)}, test: {len(self.trg_test_wsis)}"
        )
        print(
            f"[patch (target)] train: {len(trg_train_files)}, valid: {len(trg_valid_files)}, test: {len(trg_test_files)}"
        )
        print(
            f"[wsi (all)]  train: {len(self.src_train_wsis) + 1}, valid: {len(self.trg_valid_wsis)}, test: {len(self.trg_test_wsis)}"
        )
        print(
            f"[patch (all)] train: {len(train_files)}, valid: {len(valid_files)}, test: {len(test_files)}"
        )

        self.train_data = WSI(train_files, self.classes, self.shape, self.transform)

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

    def get_wsi_list(self, imgs_dir: str):
        wsi_list = []
        for i in range(len(self.sub_classes)):
            sub_cl = self.sub_classes[i]
            wsi_list.extend([p[:-4] for p in os.listdir(imgs_dir + f"{sub_cl}/")])
        wsi_list = list(set(wsi_list))
        # os.listdirによる実行時における要素の順不同対策のため
        return natsorted(wsi_list)

    def get_wsi_split(self):
        return (
            [self.trg_selected_wsi],
            [self.trg_selected_wsi],
            natsorted(self.trg_test_wsis),
        )
