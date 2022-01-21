import os
import sys
import glob
from natsort import natsorted
import re

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from S.dataset import WSI, WSIDataset


class WSIDatasetST1_TL_ValT(WSIDataset):
    """
    single target WSIを用いたTransfer Learning用
    valid dataには複数枚のtarget WSIを使用
    """

    def __init__(
        self,
        trg_train_wsi: str,
        trg_valid_wsis: list,
        trg_test_wsis: list,
        trg_imgs_dir: str,
        classes: list = [0, 1, 2, 3],
        shape: tuple = (256, 256),
        transform: dict = None,
    ):
        self.trg_train_wsi = trg_train_wsi
        self.trg_valid_wsis = trg_valid_wsis
        self.trg_test_wsis = trg_test_wsis
        self.trg_imgs_dir = trg_imgs_dir
        self.classes = classes
        self.shape = shape
        self.transform = transform
        self.sub_classes = self.get_sub_classes()

        # 各WSIのリストからパッチのパスリストを取得 (target)
        self.trg_train_files = self.get_files([self.trg_train_wsi], self.trg_imgs_dir)
        self.trg_valid_files = self.get_files(self.trg_valid_wsis, self.trg_imgs_dir)
        self.trg_test_files = self.get_files(self.trg_test_wsis, self.trg_imgs_dir)

        train_files = self.trg_train_files
        valid_files = self.trg_valid_files
        test_files = natsorted(self.trg_test_files)

        self.data_len = len(train_files) + len(valid_files) + len(test_files)
        print(
            f"[wsi (target)]  train: 1, valid: {len(self.trg_valid_wsis)}, test: {len(self.trg_test_wsis)}"
        )
        print(
            f"[wsi (all)]  train: 1, valid: {len(self.trg_valid_wsis)}, test: {len(self.trg_test_wsis)}"
        )
        print(
            f"[patch (target)] train: {len(self.trg_train_files)}, valid: {len(self.trg_valid_files)}, test: {len(self.trg_test_files)}"
        )
        print(
            f"[patch (all)] train: {len(train_files)}, valid: {len(self.trg_valid_files)}, test: {len(test_files)}"
        )

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
            natsorted([self.trg_train_wsi]),
            natsorted(self.trg_valid_wsis),
            natsorted(self.trg_test_wsis),
        )

    def wsi_num(self):
        wsi_num = len([self.trg_train_wsi])
        wsi_num += len(self.trg_valid_wsis)
        wsi_num += len(self.trg_test_wsis)
        return len(wsi_num)

    def get(self):
        return self.trg_train_data, self.valid_data, self.test_data
