
import logging
import os
import sys
import yaml
import joblib
import glob
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from natsort import natsorted
from PIL import Image
from openslide import OpenSlide

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from S.dataset import WSI
from S.util import fix_seed
from S.model import build_model


class makePredmap(object):
    def __init__(self, wsi_name, classes, level, wsi_dir, overlaid_mask_dir):
        self.wsi_name = wsi_name
        self.classes = classes

        self.wsi_dir = wsi_dir
        self.wsi_path = f"{self.wsi_dir}{self.wsi_name}.ndpi"

        self.overlaid_mask_dir = overlaid_mask_dir

        self.default_level = 5
        self.level = level
        self.length = 256
        self.resized_size = (
            int(self.length / 2 ** (self.default_level - self.level)),
            int(self.length / 2 ** (self.default_level - self.level))
        )
        self.size = (self.length, self.length)
        self.stride = 256

    def get_wsi_name(self):
        return self.wsi_name

    def num_to_color(self, num):
        if isinstance(num, list):
            num = num[0]

        if num == 0:
            color = (200, 200, 200)
        elif num == 1:
            color = (255, 40, 0)
        elif num == 2:
            color = (0, 65, 255)
        elif num == 3:
            color = (53, 161, 107)
        elif num == 4:
            color = (250, 245, 0)
        elif num == 5:
            color = (102, 204, 255)
        else:
            sys.exit("invalid number:" + str(num))
        return color

    # 予測結果からセグメンテーション画像を生成
    def preds_to_image(
        self,
        preds: list,
        output_dir: str,
        output_name: str,
        cnt=0,
    ):
        wsi = OpenSlide(self.wsi_path)

        width = wsi.level_dimensions[self.level][0]
        height = wsi.level_dimensions[self.level][1]
        row_max = int((width - self.size[0]) / self.stride + 1)
        column_max = int((height - self.size[1]) / self.stride + 1)

        canvas_shape = (
            self.resized_size[1] * column_max,
            self.resized_size[0] * row_max, 3)
        canvas_nd = np.full(canvas_shape, 255, dtype=np.uint8)

        for column in range(column_max):
            for row in range(row_max):
                y = preds[cnt].argmax(dim=0).numpy().copy()
                y_color = self.num_to_color(y)

                # canvas_nd[
                #     row * self.resized_size[0]:(row + 1) * self.resized_size[0],
                #     column * self.resized_size[1]:(column + 1) * self.resized_size[1], :] = y_color
                canvas_nd[
                    column * self.resized_size[1]:(column + 1) * self.resized_size[1],
                    row * self.resized_size[0]:(row + 1) * self.resized_size[0], :] = y_color
                cnt = cnt + 1
        canvas = Image.fromarray(canvas_nd)
        canvas.save(output_dir + output_name + ".png", "PNG", quality=100)

    # 背景&対象外領域をマスク
    def make_black_mask(self, input_dir, output_dir, suffix=None):
        if suffix is None:
            filename = self.wsi_name
        else:
            filename = self.wsi_name + suffix

        image = Image.open(
            input_dir + filename + ".png"
        )
        image_gt = Image.open(self.overlaid_mask_dir + self.wsi_name + "_overlaid.tif")

        WIDTH = image.size[0]
        HEIGHT = image.size[1]

        for x in range(WIDTH):
            for y in range(HEIGHT):
                if image_gt.getpixel((x, y)) == (0, 0, 0):
                    image.putpixel((x, y), (0, 0, 0))
                elif image_gt.getpixel((x, y)) == (255, 255, 255):
                    image.putpixel((x, y), (255, 255, 255))

        image.save(
            output_dir + filename + ".png",
            "PNG",
            quality=100,
            optimize=True,
        )


def main(
    test_set: str = "trg_unl",
    trg_l_wsi: str = "03_G144",
    l_trg_set: str = "top",
    facility: str = "MF0003",
    cv_num: int = 0,
):
    fix_seed(0)

    config_path = "../ST_MICCAI/config_st_cl[0, 1, 2]_valt20_pretrained.yaml"
    with open(config_path) as file:
        config = yaml.safe_load(file.read())

    # ========================================================== #
    MAIN_DIR = "/mnt/secssd/SSDA_Annot_WSI_strage/"

    WSI_DIR = MAIN_DIR + f"mnt1/{facility}/origin/"
    MASK_DIR = MAIN_DIR + f"mnt1/{facility}/mask_cancergrade/overlaid_{config['main']['classes']}/"

    PATCH_DIR = MAIN_DIR + f"mnt3/{facility}/"
    OUTPUT_DIR = f"{config['main']['result_dir']}predmap/l_trg_{trg_l_wsi}/cv{cv_num}/"
    os.makedirs(OUTPUT_DIR) if os.path.isdir(OUTPUT_DIR) is False else None

    # predmapを作成済みのWSIはスキップ
    skip_list = [predmap_fname.replace(".png", "") for predmap_fname in os.listdir(OUTPUT_DIR)]
    skip_list = list(set(skip_list))
    # ========================================================== #

    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    weight_list = [
        f"{config['test']['weight_dir']}{config['main']['prefix']}_{config['main']['src_facility']}_{l_trg_set}_{trg_l_wsi}_{config['main']['classes']}/" + name
        for name
        in config['test']['weight_names'][l_trg_set][trg_l_wsi]
    ]
    weight_path = weight_list[cv_num]

    wsis = joblib.load(
        config['dataset']['jb_dir']
        + f"{facility}/"
        + f"{test_set}_wsi.jb"
    )

    net = build_model(
        config['main']['model'],
        num_classes=len(config['main']['classes'])
    )

    weight_path = weight_list[cv_num]
    logging.info("Loading model {}".format(weight_path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(
        torch.load(weight_path, map_location=device))

    net.eval()
    for wsi_name in wsis:
        logging.info(f"== {wsi_name} ==")

        # 既に作成済みのwsiはスキップ
        tmp_skip_list = [s for s in skip_list if s in wsi_name]
        if len(tmp_skip_list) > 0:
            print(f"skip: {wsi_name}")
            continue

        PMAP = makePredmap(
            wsi_name=wsi_name,
            classes=config['main']['classes'],
            level=0,
            wsi_dir=WSI_DIR,
            overlaid_mask_dir=MASK_DIR
        )

        patch_list = natsorted(glob.glob(PATCH_DIR + f"/{wsi_name}/*.png", recursive=False))

        test_data = WSI(
            patch_list,
            config['main']['classes'],
            tuple(config['main']['shape']),
            transform={'Resize': True, 'HFlip': False, 'VFlip': False},
            is_pred=True
        )

        loader = DataLoader(
            test_data, batch_size=config['main']['batch_size'],
            shuffle=False, num_workers=0, pin_memory=True)

        n_val = len(loader)  # the number of batch

        all_preds = []
        logging.info("predict class...")
        with tqdm(total=n_val, desc='prediction-map', unit='batch', leave=False) as pbar:
            for batch in loader:
                imgs = batch['image']
                imgs = imgs.to(device=device, dtype=torch.float32)

                with torch.no_grad():
                    preds = net(imgs)
                preds = nn.Softmax(dim=1)(preds).to('cpu').detach()
                all_preds.extend(preds)

                pbar.update()

        # 予測結果からセグメンテーション画像を生成
        logging.info("make segmented image from prediction results ...")
        PMAP.preds_to_image(
            preds=all_preds,
            output_dir=OUTPUT_DIR,
            output_name=wsi_name
        )

        # 背景&対象外領域をマスク
        logging.info("mask bg & other classes area...")
        PMAP.make_black_mask(OUTPUT_DIR, OUTPUT_DIR)


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # cv_num = 0
    # main(test_set="trg_unl", trg_l_wsi="03_G144", l_trg_set="top", facility="MF0003", cv_num=cv_num)
    # main(test_set="trg_unl", trg_l_wsi="03_G177", l_trg_set="med", facility="MF0003", cv_num=cv_num)
    # main(test_set="trg_unl", trg_l_wsi="03_G109-1", l_trg_set="btm", facility="MF0003", cv_num=cv_num)

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # cv_num = 1
    # main(test_set="trg_unl", trg_l_wsi="03_G144", l_trg_set="top", facility="MF0003", cv_num=cv_num)
    # main(test_set="trg_unl", trg_l_wsi="03_G177", l_trg_set="med", facility="MF0003", cv_num=cv_num)
    # main(test_set="trg_unl", trg_l_wsi="03_G109-1", l_trg_set="btm", facility="MF0003", cv_num=cv_num)

    # cv_num = 2
    # main(test_set="trg_unl", trg_l_wsi="03_G144", l_trg_set="top", facility="MF0003", cv_num=cv_num)
    # main(test_set="trg_unl", trg_l_wsi="03_G177", l_trg_set="med", facility="MF0003", cv_num=cv_num)
    # main(test_set="trg_unl", trg_l_wsi="03_G109-1", l_trg_set="btm", facility="MF0003", cv_num=cv_num)

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    cv_num = 3
    main(test_set="trg_unl", trg_l_wsi="03_G144", l_trg_set="top", facility="MF0003", cv_num=cv_num)
    main(test_set="trg_unl", trg_l_wsi="03_G177", l_trg_set="med", facility="MF0003", cv_num=cv_num)
    main(test_set="trg_unl", trg_l_wsi="03_G109-1", l_trg_set="btm", facility="MF0003", cv_num=cv_num)

    cv_num = 4
    main(test_set="trg_unl", trg_l_wsi="03_G144", l_trg_set="top", facility="MF0003", cv_num=cv_num)
    main(test_set="trg_unl", trg_l_wsi="03_G177", l_trg_set="med", facility="MF0003", cv_num=cv_num)
    main(test_set="trg_unl", trg_l_wsi="03_G109-1", l_trg_set="btm", facility="MF0003", cv_num=cv_num)
