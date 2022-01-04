import os
import sys
import re
import random
import glob
from natsort import natsorted

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from S.dataset import WSI, WSIDataset


class WSIDataset_ST1_ADA_ValT(WSIDataset):
    """
    Adversarial Domain Adaptation用
    labeled sourceのWSIとsingle labeled target WSI, unlabeled target WSIを使用
    valid dataには複数枚のtarget WSIを使用
    """

    def __init__(
        self,
        l_src_train_wsis: list,
        l_trg_train_wsi: str,
        unl_trg_train_wsis: list,
        trg_valid_wsis: list,
        trg_test_wsis: list,
        src_imgs_dir: str,
        trg_imgs_dir: str,
        classes: list = [0, 1, 2],
        shape: tuple = (256, 256),
        transform: dict = None,
        balance_domain: bool = False,
    ):
        self.l_src_train_wsis = l_src_train_wsis
        self.l_trg_train_wsi = l_trg_train_wsi
        self.unl_trg_train_wsis = unl_trg_train_wsis
        self.trg_valid_wsis = trg_valid_wsis
        self.trg_test_wsis = trg_test_wsis
        self.src_imgs_dir = src_imgs_dir
        self.trg_imgs_dir = trg_imgs_dir
        self.classes = classes
        self.shape = shape
        self.transform = transform
        self.sub_classes = self.get_sub_classes()

        # tarin用のWSIのリストからpatchのpathリストを取得 (source)
        self.l_src_train_files = self.get_files(self.l_src_train_wsis, self.src_imgs_dir)

        # 各WSIのリストからpatchのpathリストを取得 (target)
        self.l_trg_train_files = self.get_files([self.l_trg_train_wsi], self.trg_imgs_dir)
        self.unl_trg_train_files = self.get_files(self.unl_trg_train_wsis, self.trg_imgs_dir)
        self.trg_valid_files = self.get_files(self.trg_valid_wsis, self.trg_imgs_dir)
        self.trg_test_files = self.get_files(self.trg_test_wsis, self.trg_imgs_dir)

        # FIXME: Debug用 ------------ #
        # self.l_src_train_files = self.l_src_train_files[:512]
        # # self.l_trg_train_files = self.l_trg_train_files[:512]
        # self.unl_trg_train_files = self.unl_trg_train_files[:512]
        # -------------------------- #

        # l_src_train_filesとl_trg_train_filesを同数にする
        if balance_domain:
            self.rebalance()

        train_files = self.l_src_train_files + self.l_trg_train_files + self.unl_trg_train_files
        valid_files = self.trg_valid_files
        test_files = natsorted(self.trg_test_files)

        self.data_len = len(train_files) + len(valid_files) + len(test_files)
        print(f"[wsi (source)]  train: (l) {len(self.l_src_train_wsis)}")
        print(
            f"[wsi (target)]  train: (l) 1, (unl) {len(self.unl_trg_train_wsis)}, valid: {len(self.trg_valid_wsis)}, test: {len(self.trg_test_wsis)}"
        )
        print(
            f"[wsi (all)]  train: {len(self.l_src_train_wsis) + len(self.unl_trg_train_wsis) + 1}, valid: {len(self.trg_valid_wsis)}, test: {len(self.trg_test_wsis)}"
        )
        print(f"[patch (source)] train: {len(self.l_src_train_files)}")
        print(
            f"[patch (target)] train: (l) {len(self.l_trg_train_files)}, (unl) {len(self.unl_trg_train_files)}, valid: {len(self.trg_valid_files)}, test: {len(self.trg_test_files)}"
        )
        print(
            f"[patch (all)] train: {len(train_files)}, valid: {len(self.trg_valid_files)}, test: {len(test_files)}"
        )

        self.l_src_train_data = WSI(self.l_src_train_files, self.classes, self.shape, self.transform)
        self.l_trg_train_data = WSI(self.l_trg_train_files, self.classes, self.shape, self.transform)
        self.unl_trg_train_data = WSI(self.unl_trg_train_files, self.classes, self.shape, self.transform)

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
            natsorted(self.l_src_train_wsis + [self.l_trg_train_wsi] + self.unl_trg_train_wsis),
            natsorted(self.trg_valid_wsis),
            natsorted(self.trg_test_wsis),
        )

    def wsi_num(self):
        wsi_num = len(self.l_src_train_wsis + [self.l_trg_train_wsi] + self.unl_trg_train_wsis)
        wsi_num += len(self.trg_valid_wsis)
        wsi_num += len(self.trg_test_wsis)
        return len(wsi_num)

    def rebalance(self):
        """
        labeled sourceとlabeled targetでパッチ数を同数になるよう増量
        """
        random.seed(0)
        l_src_train_num = len(self.l_src_train_files)
        l_trg_train_num = len(self.l_trg_train_files)
        if l_src_train_num > l_trg_train_num:  # targetの方が少ない場合
            add_num = l_src_train_num - l_trg_train_num
            self.l_trg_train_files += \
                random.choices(self.l_trg_train_files, k=add_num)
        else:  # sourceの方が少ない場合
            add_num = l_trg_train_num - l_src_train_num
            self.l_src_train_files += \
                random.choices(self.l_src_train_files, k=add_num)

        assert len(self.l_src_train_files) == len(self.l_trg_train_files), \
            f"l_src_train_files: {len(self.l_src_train_files)}, l_trg_train_files: {len(self.l_trg_train_files)}"

    def get(self):
        return self.l_src_train_data, self.l_trg_train_data, self.unl_trg_train_data, self.valid_data, self.test_data
