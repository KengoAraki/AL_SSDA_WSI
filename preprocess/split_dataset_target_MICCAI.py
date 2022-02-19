import os
import joblib
import logging
import glob
import copy
from natsort import natsorted
import re
import random
import pandas as pd


# ========================= #
#    For SSDA Target
# ========================= #
class SSDATargetDatasetCE(object):
    def __init__(
        self,
        trg_l_top_wsis: list,
        trg_l_med_wsis: list,
        trg_l_btm_wsis: list,
        imgs_dir: str,
        valid_wsi_num: int = 20,
        classes: list = [0, 1, 2, 3]
    ):
        self.trg_l_top_wsis = trg_l_top_wsis
        self.trg_l_med_wsis = trg_l_med_wsis
        self.trg_l_btm_wsis = trg_l_btm_wsis
        self.imgs_dir = imgs_dir
        self.valid_wsi_num = valid_wsi_num
        self.classes = classes
        self.sub_classes = self.get_sub_classes()

        # # Targetの対象クラスを含むWSIのリストを取得
        # self.all_wsis = []
        # for i in range(len(self.sub_classes)):
        #     sub_cl = self.sub_classes[i]
        #     self.all_wsis.extend([p[:-4] for p in os.listdir(self.imgs_dir + f"{sub_cl}/")])
        # self.all_wsis = list(set(self.all_wsis))
        # # os.listdirによる実行時における要素の順不同対策のため
        # self.all_wsis = natsorted(self.all_wsis)

        # ===== 実験条件を修論と揃えるため ===== #
        jb_dir = "/mnt/secssd/SSDA_Annot_WSI_strage/dataset/"
        facility = "MF0003"
        cv_num = 0
        self.all_wsis = joblib.load(
            jb_dir + f"{facility}/" + f"cv{cv_num}_train_" + f"{facility}_wsi.jb"
        )
        self.all_wsis += joblib.load(
            jb_dir + f"{facility}/" + f"cv{cv_num}_valid_" + f"{facility}_wsi.jb"
        )
        self.all_wsis += joblib.load(
            jb_dir + f"{facility}/" + f"cv{cv_num}_test_" + f"{facility}_wsi.jb"
        )
        self.all_wsis = natsorted(self.all_wsis)
        # ============================== #

        self.trg_unl_wsis = copy.deepcopy(self.all_wsis)
        # labeled target用WSIを取り除く
        for wsi in self.trg_l_top_wsis:
            self.trg_unl_wsis.remove(wsi)
        for wsi in self.trg_l_med_wsis:
            self.trg_unl_wsis.remove(wsi)
        for wsi in self.trg_l_btm_wsis:
            self.trg_unl_wsis.remove(wsi)

        # valid_wsi_numの数だけランダムに割当
        random.shuffle(self.trg_unl_wsis)
        self.valid_wsis = natsorted(self.trg_unl_wsis[:self.valid_wsi_num])

        # targetのvalid用WSIを取り除く
        for wsi in self.valid_wsis:
            self.trg_unl_wsis.remove(wsi)

        self.trg_unl_wsis = natsorted(self.trg_unl_wsis)

    def __len__(self):
        return len(self.all_wsis)

    def get_wsis(self):
        return self.trg_l_wsis, self.trg_unl_wsis, self.valid_wsis

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
        re_pattern = re.compile('|'.join([f"/{i}/" for i in self.sub_classes]))

        files_list = []
        for wsi in wsis:
            files_list.extend(
                [
                    p for p in glob.glob(self.imgs_dir + f"*/{wsi}_*/*.png", recursive=True)
                    if bool(re_pattern.search(p))
                ]
            )
        return files_list


def save_SSDA_target_dataset(
    trg_l_top_wsis: list,
    trg_l_med_wsis: list,
    trg_l_btm_wsis: list,
    valid_wsi_num: int,
    classes: list,
    imgs_dir: str,
    output_dir: str
):
    dataset = SSDATargetDatasetCE(
        trg_l_top_wsis=trg_l_top_wsis,
        trg_l_med_wsis=trg_l_med_wsis,
        trg_l_btm_wsis=trg_l_btm_wsis,
        imgs_dir=imgs_dir,
        valid_wsi_num=valid_wsi_num,
        classes=classes
    )

    trg_unl_wsis = dataset.trg_unl_wsis
    valid_wsis = dataset.valid_wsis

    trg_l_top_files = dataset.get_files(trg_l_top_wsis)
    trg_l_med_files = dataset.get_files(trg_l_med_wsis)
    trg_l_btm_files = dataset.get_files(trg_l_btm_wsis)
    trg_unl_files = dataset.get_files(trg_unl_wsis)
    valid_files = dataset.get_files(valid_wsis)

    logging.info(f"[wsi]   trg_l(top): {len(trg_l_top_wsis)}, trg_l(med): {len(trg_l_med_wsis)}, trg_l(btm): {len(trg_l_btm_wsis)}, trg_unl: {len(trg_unl_wsis)}, valid: {len(valid_wsis)}")
    logging.info(f"[patch] trg_l(top): {len(trg_l_top_files)}, trg_l(med): {len(trg_l_med_files)}, trg_l(btm): {len(trg_l_btm_files)},  trg_unl: {len(trg_unl_files)}, valid: {len(valid_files)}")

    # WSI割当のリストを保存
    joblib.dump(trg_l_top_wsis, output_dir + "trg_l_top_wsi.jb", compress=3)
    joblib.dump(trg_l_med_wsis, output_dir + "trg_l_med_wsi.jb", compress=3)
    joblib.dump(trg_l_btm_wsis, output_dir + "trg_l_btm_wsi.jb", compress=3)
    joblib.dump(trg_unl_wsis, output_dir + "trg_unl_wsi.jb", compress=3)
    joblib.dump(valid_wsis, output_dir + "valid_wsi.jb", compress=3)

    # # 各データのリスト(path)を保存
    # joblib.dump(trg_l_top_files, output_dir + "trg_l_top.jb", compress=3)
    # joblib.dump(trg_l_med_files, output_dir + "trg_l_med.jb", compress=3)
    # joblib.dump(trg_l_btm_files, output_dir + "trg_l_btm.jb", compress=3)
    # joblib.dump(trg_unl_files, output_dir + "trg_unl.jb", compress=3)
    # joblib.dump(valid_files, output_dir + "valid.jb", compress=3)

    with open(output_dir + "SSDA_target_dataset_MICCAI.txt", mode='w') as f:
        f.write(
            "== [wsi] ==\n"
            + f"trg_l(top): {len(trg_l_top_wsis)}, trg_l(med): {len(trg_l_med_wsis)}, trg_l(btm): {len(trg_l_btm_wsis)},\n trg_unl: {len(trg_unl_wsis)}, valid: {len(valid_wsis)}"
            + "\n==============\n")
        f.write(
            "\n== [patch] ==\n"
            + f"trg_l(top): {len(trg_l_top_files)}, trg_l(med): {len(trg_l_med_files)}, trg_l(btm): {len(trg_l_btm_files)},\n trg_unl: {len(trg_unl_files)}, valid: {len(valid_files)}"
            + "\n==============\n")

        f.write("\n== trg_l_top (wsi) ==\n")
        for i in range(len(trg_l_top_wsis)):
            f.write(f"{trg_l_top_wsis[i]}\n")

        f.write("\n== trg_l_med (wsi) ==\n")
        for i in range(len(trg_l_med_wsis)):
            f.write(f"{trg_l_med_wsis[i]}\n")

        f.write("\n== trg_l_btm (wsi) ==\n")
        for i in range(len(trg_l_btm_wsis)):
            f.write(f"{trg_l_btm_wsis[i]}\n")

        f.write("\n== trg_unl (wsi) ==\n")
        for i in range(len(trg_unl_wsis)):
            f.write(f"{trg_unl_wsis[i]}\n")

        f.write("\n== valid (wsi) ==\n")
        for i in range(len(valid_wsis)):
            f.write(f"{valid_wsis[i]}\n")


def get_l_trg_wsis(
    csv_path: str,
):
    df = pd.read_csv(csv_path)

    # sample_numが0のWSIを除去
    print(f"bf: {len(df)}")
    df = df[df['sample_num'] > 0]
    print(f"af: {len(df)}")

    df_sort = df.sort_values('entropy', ascending=False)  # cluster-entropyの大きい順にソート
    wsi_list_descsort = df_sort['wsi'].tolist()
    entropy_list_descsort = df_sort['entropy'].tolist()

    print(f"[max]: {wsi_list_descsort[0]} ({entropy_list_descsort[0]})")
    print(f"[med]: {wsi_list_descsort[len(wsi_list_descsort) // 2]} ({entropy_list_descsort[len(wsi_list_descsort) // 2]})")
    print(f"[min]: {wsi_list_descsort[-1]} ({entropy_list_descsort[-1]})")

    top_wsis = wsi_list_descsort[:5]
    med_idx = len(wsi_list_descsort) // 2
    med_wsis = wsi_list_descsort[med_idx - 2: med_idx + 3]
    btm_wsis = wsi_list_descsort[-5:]

    print(f"[top]: {top_wsis}")
    print(f"[med]: {med_wsis}")
    print(f"[btm]: {btm_wsis}")
    return top_wsis, med_wsis, btm_wsis


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    imgs_dir = "/mnt/secssd/SSDA_Annot_WSI_strage/mnt2/MF0003/"
    output_dir = "/mnt/secssd/AL_SSDA_WSI_MICCAI_strage/dataset/MF0003/"
    csv_path = "/home/kengoaraki/Project/DA/AL_SSDA_WSI/analysis/MF0003_cluster_entropy_all.csv"

    trg_l_top_wsis, trg_l_med_wsis, trg_l_btm_wsis = get_l_trg_wsis(csv_path)

    classes = [0, 1, 2]
    valid_wsi_num = 20

    save_SSDA_target_dataset(
        trg_l_top_wsis=trg_l_top_wsis,
        trg_l_med_wsis=trg_l_med_wsis,
        trg_l_btm_wsis=trg_l_btm_wsis,
        valid_wsi_num=valid_wsi_num,
        classes=classes,
        imgs_dir=imgs_dir,
        output_dir=output_dir
    )
