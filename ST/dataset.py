import os
import sys
import glob
from natsort import natsorted
import re
import random

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from S.dataset import WSI, WSIDataset


class WSIDatasetST1_ValT(WSIDataset):
    """
    sourceのWSIとsingle target WSIを用いたFine-tuning 用
    valid dataには複数枚のtarget WSIを使用
    """

    def __init__(
        self,
        trg_train_wsi: str,
        src_train_wsis: list,
        trg_valid_wsis: list,
        trg_test_wsis: list,
        src_imgs_dir: str,
        trg_imgs_dir: str,
        classes: list = [0, 1, 2, 3],
        shape: tuple = (256, 256),
        transform: dict = None,
        balance_domain: bool = False,
    ):
        self.trg_train_wsi = trg_train_wsi
        self.src_train_wsis = src_train_wsis
        self.trg_valid_wsis = trg_valid_wsis
        self.trg_test_wsis = trg_test_wsis
        self.src_imgs_dir = src_imgs_dir
        self.trg_imgs_dir = trg_imgs_dir
        self.classes = classes
        self.shape = shape
        self.transform = transform
        self.sub_classes = self.get_sub_classes()

        # tarin用のWSIのリストからパッチのパスリストを取得 (source)
        self.src_train_files = self.get_files(self.src_train_wsis, self.src_imgs_dir)

        # 各WSIのリストからパッチのパスリストを取得 (target)
        self.trg_train_files = self.get_files([self.trg_train_wsi], self.trg_imgs_dir)
        self.trg_valid_files = self.get_files(self.trg_valid_wsis, self.trg_imgs_dir)
        self.trg_test_files = self.get_files(self.trg_test_wsis, self.trg_imgs_dir)

        # src_train_filesとtrg_train_filesを同数にする
        if balance_domain:
            self.rebalance()

        train_files = self.src_train_files + self.trg_train_files
        valid_files = self.trg_valid_files
        test_files = natsorted(self.trg_test_files)

        self.data_len = len(train_files) + len(valid_files) + len(test_files)
        print(f"[wsi (source)]  train: {len(self.src_train_wsis)}")
        print(
            f"[wsi (target)]  train: 1, valid: {len(self.trg_valid_wsis)}, test: {len(self.trg_test_wsis)}"
        )
        print(
            f"[wsi (all)]  train: {len(self.src_train_wsis) + 1}, valid: {len(self.trg_valid_wsis)}, test: {len(self.trg_test_wsis)}"
        )
        print(f"[patch (source)] train: {len(self.src_train_files)}")
        print(
            f"[patch (target)] train: {len(self.trg_train_files)}, valid: {len(self.trg_valid_files)}, test: {len(self.trg_test_files)}"
        )
        print(
            f"[patch (all)] train: {len(train_files)}, valid: {len(self.trg_valid_files)}, test: {len(test_files)}"
        )

        self.src_train_data = WSI(self.src_train_files, self.classes, self.shape, self.transform)
        self.trg_train_data = WSI(self.trg_train_files, self.classes, self.shape, self.transform)

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
            natsorted(self.src_train_wsis + [self.trg_train_wsi]),
            natsorted(self.trg_valid_wsis),
            natsorted(self.trg_test_wsis),
        )

    def wsi_num(self):
        wsi_num = len(self.src_train_wsis + [self.trg_train_wsi])
        wsi_num += len(self.trg_valid_wsis)
        wsi_num += len(self.trg_test_wsis)
        return len(wsi_num)

    def rebalance(self):
        random.seed(0)
        src_train_num = len(self.src_train_files)
        trg_train_num = len(self.trg_train_files)
        if src_train_num > trg_train_num:  # targetの方が少ない場合
            add_num = src_train_num - trg_train_num
            self.trg_train_files += \
                random.choices(self.trg_train_files, k=add_num)
        else:  # sourceの方が少ない場合
            add_num = trg_train_num - src_train_num
            self.src_train_files += \
                random.choices(self.src_train_files, k=add_num)

        assert len(self.src_train_files) == len(self.trg_train_files), \
            f"src_train_files: {len(self.src_train_files)}, trg_train_files: {len(self.trg_train_files)}"

    def get(self):
        return self.src_train_data, self.trg_train_data, self.valid_data, self.test_data


if __name__ == '__main__':
    import yaml
    import joblib

    config_path = "./ST/config_st_cl[0, 1, 2]_valt3.yaml"

    with open(config_path) as file:
        config = yaml.safe_load(file.read())

    input_shape = tuple(config['main']['shape'])
    transform = {"Resize": True, "HFlip": True, "VFlip": True}

    # WSIのリストを取得 (target)
    trg_train_wsis = joblib.load(
        config['main']['jb_dir']
        + f"{config['main']['trg_facility']}/"
        + "trg_l_wsi.jb"
    )
    trg_valid_wsis = joblib.load(
        config['main']['jb_dir']
        + f"{config['main']['trg_facility']}/"
        + "valid_wsi.jb"
    )
    trg_test_wsis = joblib.load(
        config['main']['jb_dir']
        + f"{config['main']['trg_facility']}/"
        + "trg_unl_wsi.jb"
    )

    # WSIのリストを取得 (source)
    cv_num = 0
    src_train_wsis = joblib.load(
        config['main']['jb_dir']
        + f"{config['main']['src_facility']}/"
        + f"cv{cv_num}_"
        + f"train_{config['main']['src_facility']}_wsi.jb"
    )

    for trg_selected_wsi in trg_train_wsis:
        print(f"===== {trg_selected_wsi} =====")
        dataset = WSIDatasetST1_ValT(
            trg_train_wsi=trg_selected_wsi,
            src_train_wsis=src_train_wsis,
            trg_valid_wsis=trg_valid_wsis,
            trg_test_wsis=trg_test_wsis,
            src_imgs_dir=config['dataset']['src_imgs_dir'],
            trg_imgs_dir=config['dataset']['trg_imgs_dir'],
            classes=config['main']['classes'],
            shape=input_shape,
            transform=transform,
            balance_domain=False,
        )
        print(len(dataset))
