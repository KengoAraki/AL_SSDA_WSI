import logging
import os
import sys
import yaml
import joblib
import glob
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from natsort import natsorted

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from S.dataset import WSI
from S.util import fix_seed
from S.model import build_model
from S.predict import makePredmap


def main():
    fix_seed(0)

    config_path = "../ST/config_st_cl[0, 1, 2]_valt3_pretrained.yaml"

    with open(config_path) as file:
        config = yaml.safe_load(file.read())

    # ========================================================== #
    is_likelihood = False
    facility = "MF0003"
    test_set = "trg_unl"
    trg_l_wsi = "03_G144"
    cv_num = 2
    weight_path = (
        config['test']['weight_dir'][trg_l_wsi]
        + config['test']['weight_names'][trg_l_wsi][cv_num]
    )

    MAIN_DIR = "/mnt/secssd/SSDA_Annot_WSI_strage/"

    WSI_DIR = MAIN_DIR + f"mnt1/{facility}/origin/"
    MASK_DIR = (
        MAIN_DIR
        + f"mnt1/{facility}/mask_cancergrade/overlaid_{config['main']['classes']}/"
    )

    PATCH_DIR = MAIN_DIR + f"mnt3/{facility}/"
    PRED_DIR = MAIN_DIR + f"mnt4/st_pretrained_{facility}_{trg_l_wsi}/"
    OUTPUT_DIR = f"/mnt/secssd/AL_SSDA_WSI_strage/st_pretrained_result/test/{trg_l_wsi}/predmap/"
    # ========================================================== #

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    test_wsis = joblib.load(
        config['dataset']['jb_dir']
        + f"{facility}/"
        + f"{test_set}_wsi.jb"
    )

    net = build_model(
        config["main"]["model"], num_classes=len(config["main"]["classes"])
    )

    logging.info("Loading model {}".format(weight_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")
    net.to(device=device)
    net.load_state_dict(torch.load(weight_path, map_location=device))

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
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
