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

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from S.dataset import WSI
from S.util import fix_seed
from S.model import build_model
from preprocess.openslide_wsi import OpenSlideWSI

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class makePredmap(object):
    def __init__(self, wsi_name, classes, wsi_dir, overlaid_mask_dir):
        self.wsi_name = wsi_name
        self.classes = classes

        self.wsi_dir = wsi_dir
        self.wsi_path = f"{self.wsi_dir}{self.wsi_name}.ndpi"

        self.overlaid_mask_dir = overlaid_mask_dir

        self.default_level = 5
        self.level = 0
        self.length = 256
        self.resized_size = (
            int(self.length / 2 ** (self.default_level - self.level)),
            int(self.length / 2 ** (self.default_level - self.level)),
        )
        self.size = (self.length, self.length)
        self.stride = 256

    def make_output_dir(self, main_dir, wsi_name):
        output_dir = f"{main_dir}{wsi_name}/"
        os.makedirs(output_dir) if os.path.isdir(output_dir) is False else None
        return output_dir

    def get_wsi_name(self):
        return self.wsi_name

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

    # 予測したパッチの着色
    def color_patch(self, y_classes, test_data_list, output_dir):
        for y, test_data in zip(y_classes, test_data_list):
            y = y.argmax(dim=0).numpy().copy()  # yはargmax前のsoftmax出力
            filename, _ = os.path.splitext(os.path.basename(test_data))
            canvas = np.zeros((256, 256, 3))
            for cl in range(len(self.classes)):
                canvas[y == cl] = self.num_to_color(self.classes[cl])
            canvas = Image.fromarray(np.uint8(canvas))
            canvas.save(
                output_dir + filename + ".png", "PNG", quality=100, optimize=True
            )

    # 予測したパッチをlikelihood-map用に着色
    def color_likelihood_patch(self, y_classes, test_data_list, output_dir):
        for y, test_data in zip(y_classes, test_data_list):
            y = y.numpy().copy()
            filename, _ = os.path.splitext(os.path.basename(test_data))
            for cl in range(len(self.classes)):
                color = self.num_to_color(self.classes[cl])
                patch_color = np.uint8(np.multiply(y[cl], color))
                canvas = np.full((256, 256, 3), patch_color)
                canvas = Image.fromarray(canvas)
                canvas.save(
                    f"{output_dir}{filename}_cl{cl}.png",
                    "PNG",
                    quality=100,
                    optimize=True,
                )

    # パッチの結合
    def merge_patch(self, patch_dir, output_dir, suffix=None):
        img = OpenSlideWSI(self.wsi_path)
        img.patch_to_image(
            self.resized_size,
            self.level,
            self.size,
            self.stride,
            input_dir=patch_dir,
            output_dir=output_dir,
            output_name=self.wsi_name,
            suffix=suffix,
            cnt=0,
        )

    # 背景&対象外領域をマスク
    def make_black_mask(self, input_dir, output_dir, suffix=None):
        if suffix is None:
            filename = self.wsi_name
        else:
            filename = self.wsi_name + suffix

        image = Image.open(input_dir + filename + ".png")
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
            output_dir + filename + ".png", "PNG", quality=100, optimize=True,
        )


def main():
    fix_seed(0)
    # config_path = './config/config_src_for_ada.yaml'
    config_path = "../config/config_src_for_ada.yaml"

    with open(config_path) as file:
        config = yaml.safe_load(file.read())

    # # ========================================================== #
    # is_likelihood = True
    # MAIN_DIR = "/mnt/ssdsub1/DFBConv_strage/"

    # WSI_DIR = MAIN_DIR + "mnt1/MF0012/origin/"
    # MASK_DIR = MAIN_DIR + f"mnt1/MF0012/mask_cancergrade/overlaid_{config['main']['classes']}/"

    # # For Baseline method
    # PATCH_DIR = MAIN_DIR + "mnt4(ForPredmap)/MF0012/"
    # PRED_DIR = MAIN_DIR + "mnt5(pred_patch)/MF0012/"
    # # OUTPUT_DIR = MAIN_DIR + "tmp/predmap/"
    # OUTPUT_DIR = MAIN_DIR + "tmp/predmap_ADA(src-MF0012)_cl[2, [1, 3]]_cv0(test)/"
    # # ========================================================== #

    # ========================================================== #
    is_likelihood = True
    MAIN_DIR = "/mnt/ssdsub1/ADA_strage/"

    wsi_name = "MF0003"
    WSI_DIR = MAIN_DIR + f"mnt1/{wsi_name}/origin/"
    MASK_DIR = (
        MAIN_DIR
        + f"mnt1/{wsi_name}/mask_cancergrade/overlaid_{config['main']['classes']}/"
    )

    # For Baseline method
    PATCH_DIR = MAIN_DIR + f"mnt4/{wsi_name}/"
    PRED_DIR = MAIN_DIR + f"mnt5/{wsi_name}/0/"
    # OUTPUT_DIR = MAIN_DIR + "tmp/predmap/"
    OUTPUT_DIR = (
        MAIN_DIR
        + "tmp/predmap_Baseline(src-MF0012, cv1_model)_trg-MF0003_cl[1, 2]_cv0(test)/"
    )
    # ========================================================== #

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    project = (
        config["main"]["model"]
        + "_"
        + config["main"]["optim"]
        + "_batch"
        + str(config["main"]["batch_size"])
        + "_shape"
        + str(config["main"]["shape"])
    )

    test_wsis = joblib.load(
        config["test"]["jb_dir"]
        + f"cv{config['test']['cv_num']}_"
        + f"{config['test']['target_data']}_{config['test']['mode']}-{config['test']['wsi']}_wsi.jb"
    )

    net = build_model(
        config["main"]["model"], num_classes=len(config["main"]["classes"])
    )

    logging.info("Loading model {}".format(config["test"]["model_path"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")
    net.to(device=device)
    net.load_state_dict(torch.load(config["test"]["model_path"], map_location=device))

    logging.info("Model loaded !")

    net.eval()
    for wsi in test_wsis:
        logging.info(f"== {wsi} ==")
        PMAP = makePredmap(
            wsi, config["main"]["classes"], wsi_dir=WSI_DIR, overlaid_mask_dir=MASK_DIR
        )

        patch_list = natsorted(glob.glob(PATCH_DIR + f"/{wsi}/*.png", recursive=False))

        test_data = WSI(
            patch_list,
            config["main"]["classes"],
            tuple(config["main"]["shape"]),
            transform={"Resize": True, "HFlip": False, "VFlip": False},
            is_pred=True,
        )

        loader = DataLoader(
            test_data,
            batch_size=config["main"]["batch_size"],
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        n_val = len(loader)  # the number of batch

        all_preds = []
        logging.info("predict class...")
        with tqdm(
            total=n_val, desc="prediction-map", unit="batch", leave=False
        ) as pbar:
            for batch in loader:
                imgs = batch["image"]
                imgs = imgs.to(device=device, dtype=torch.float32)

                with torch.no_grad():
                    preds = net(imgs)
                preds = nn.Softmax(dim=1)(preds).to("cpu").detach()
                all_preds.extend(preds)

                pbar.update()

        # 予測結果の着色パッチを作成
        logging.info("make color patch...")
        pred_out_dir = PMAP.make_output_dir(PRED_DIR, wsi)
        PMAP.color_patch(all_preds, patch_list, pred_out_dir)

        # 着色パッチを結合
        logging.info("merge color patch...")
        PMAP.merge_patch(pred_out_dir, OUTPUT_DIR)

        # 背景&対象外領域をマスク
        logging.info("mask bg & other classes area...")
        PMAP.make_black_mask(OUTPUT_DIR, OUTPUT_DIR)

        # likelihood mapの作成
        if is_likelihood:
            # 予測結果の着色パッチを作成
            logging.info("make color patch (likelihood)...")
            pred_out_dir = PMAP.make_output_dir(PRED_DIR, wsi)
            PMAP.color_likelihood_patch(all_preds, patch_list, pred_out_dir)

            for cl in range(len(config["main"]["classes"])):
                # 着色パッチを結合
                logging.info("merge color patch (likelihood)...")
                PMAP.merge_patch(pred_out_dir, OUTPUT_DIR, suffix=f"_cl{cl}")

                # 背景&対象外領域をマスク
                logging.info("mask bg & other classes area (likelihood)...")
                PMAP.make_black_mask(OUTPUT_DIR, OUTPUT_DIR, suffix=f"_cl{cl}")


if __name__ == "__main__":
    main()
