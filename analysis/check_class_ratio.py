import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import joblib
import yaml


class CheckClassRatio(object):
    def __init__(self, wsi_list, classes, mask_dir, mask_ext="_overlaid.tif"):
        self.wsi_list = wsi_list
        self.classes = self.get_sub_classes(classes)
        self.mask_dir = mask_dir
        self.mask_ext = mask_ext

    def get_sub_classes(self, classes):
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

    def draw_pie(self, x, label, colors, title=""):
        figure = plt.figure(figsize=(8, 8))
        plt.title(title)
        plt.pie(
            x,
            labels=label,
            counterclock=False,
            startangle=90,
            autopct="%.1f%%",
            wedgeprops={'linewidth': 0},
            textprops={'weight': "bold"},
            colors=colors
        )
        return figure

    # wsi (mask_cancergrade_gray) からクラスのタグを取得
    # mask: ndarray
    def get_wsi_tags(self, mask_nd):
        uniq, counts = np.unique(mask_nd, return_counts=True)
        uniq_l, counts_l = [], []
        for u, c in zip(uniq.tolist(), counts.tolist()):
            if u in self.classes:
                uniq_l.append(u)
                counts_l.append(c)
        return uniq_l, counts_l

    def num_to_color(self, num):
        if isinstance(num, list):
            num = num[0]

        if num == 0:
            color = (200, 200, 200)
        elif num == 1:
            color = (255, 0, 0)
        elif num == 2:
            color = (255, 255, 0)
        elif num == 3:
            color = (0, 255, 0)
        elif num == 4:
            color = (0, 255, 255)
        elif num == 5:
            color = (0, 0, 255)
        elif num == 6:
            color = (255, 0, 255)
        elif num == 7:
            color = (128, 0, 0)
        elif num == 8:
            color = (128, 128, 0)
        elif num == 9:
            color = (0, 128, 0)
        elif num == 10:
            color = (0, 0, 128)
        elif num == 11:
            color = (64, 64, 64)
        else:
            sys.exit("invalid number:" + str(num))
        return color

    def draw_class_ratio(self, output_dir, title):
        # 初期化
        uniq_dict, counts_dict = {}, {}
        colors = []
        for cl in self.classes:
            uniq_dict[str(cl)] = 0
            counts_dict[str(cl)] = 0
            colors.append(tuple(map(lambda x: x / 255, self.num_to_color(cl))))

        for wsi in self.wsi_list:
            mask = cv2.imread(
                self.mask_dir
                + wsi
                + self.mask_ext,
                cv2.IMREAD_GRAYSCALE
            )
            uniq, counts = self.get_wsi_tags(mask)
            for idx, u in enumerate(uniq):
                uniq_dict[str(u)] += 1
                counts_dict[str(u)] += counts[idx]

        print(f"uniq: {uniq_dict}\ncounts: {counts_dict}")

        fig = self.draw_pie(list(uniq_dict.values()), list(uniq_dict.keys()), colors, title=title)
        fig.savefig(output_dir + title + ".png", dpi=200)

        fig = self.draw_pie(list(counts_dict.values()), list(counts_dict.keys()), colors, title="[patch num] " + title)
        fig.savefig(output_dir + title + "_patch.png", dpi=200)

        plt.clf()
        plt.close()


def main():
    config_path = './config/config.yaml'
    # config_path = '../config/config.yaml'

    wsi = "MF0012"
    mode = "source"
    MASK_DIR = f"/mnt/ssdsub1/ADA_strage/mnt1/{wsi}/mask_cancergrade_gray/overlaid_[0, 1, 2, 3, 4]/"
    OUTPUT_DIR = "/mnt/ssdsub1/ADA_strage/result/dataset/"

    with open(config_path) as file:
        config = yaml.safe_load(file.read())

    classes = config['main']['classes']
    cv_num = config['main']['cv_num']
    jb_dir = config['dataset']['jb_dir']

    train_wsi = joblib.load(
        jb_dir
        + f"{mode}_{wsi}/"
        + f"cv{cv_num}_train_{mode}-{wsi}_wsi.jb"
    )

    valid_wsi = joblib.load(
        jb_dir
        + f"{mode}_{wsi}/"
        + f"cv{cv_num}_valid_{mode}-{wsi}_wsi.jb"
    )

    test_wsi = joblib.load(
        jb_dir
        + f"{mode}_{wsi}/"
        + f"cv{cv_num}_test_{mode}-{wsi}_wsi.jb"
    )

    print("== WSI nums ==")
    print(f"[wsi] train: {len(train_wsi)}, valid: {len(valid_wsi)}, test: {len(test_wsi)}")

    print("== train dataset ==")
    CD = CheckClassRatio(train_wsi, classes, MASK_DIR, mask_ext="_overlaid.tif")
    CD.draw_class_ratio(OUTPUT_DIR, f"cv{cv_num}_{wsi}_class-ratio(train)")

    print("== valid dataset ==")
    CD = CheckClassRatio(valid_wsi, classes, MASK_DIR, mask_ext="_overlaid.tif")
    CD.draw_class_ratio(OUTPUT_DIR, f"cv{cv_num}_{wsi}_class-ratio(valid)")

    print("== test dataset ==")
    CD = CheckClassRatio(test_wsi, classes, MASK_DIR, mask_ext="_overlaid.tif")
    CD.draw_class_ratio(OUTPUT_DIR, f"cv{cv_num}_{wsi}_class-ratio(test)")

    CD = CheckClassRatio(train_wsi + valid_wsi + test_wsi, classes, MASK_DIR, mask_ext="_overlaid.tif")
    CD.draw_class_ratio(OUTPUT_DIR, f"{wsi}_class-ratio(all)")


if __name__ == "__main__":
    main()
