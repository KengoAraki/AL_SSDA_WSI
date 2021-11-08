import logging
import os
import sys
import yaml
import joblib
import re
import numpy as np
from natsort import natsorted
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src_s_t1.dataset import WSI, get_files
from src_s_t1.eval import eval_net_test, plot_confusion_matrix, eval_metrics
from src_s_t1.util import fix_seed
from src_s_t1.model import build_model


def main_src():
    """
    sourceのみで訓練されたモデルで推測
    """
    fix_seed(0)
    # config_path = "../config/config_st1_cl[0, 1, 2]_val-t3.yaml"
    config_path = "../config/config_t1_cl[0, 1, 2]_val-t3.yaml"

    with open(config_path) as file:
        config = yaml.safe_load(file.read())

    weight_dir = (
        config["main"]["pretrained_weight_dir"]
        + f"{config['main']['src_facility']}_"
        + f"{config['main']['classes']}/"
    )
    weight_list = [
        weight_dir + name for name in config["main"]["pretrained_weight_names"]
    ]

    logging.basicConfig(
        level=logging.INFO,
        filename=f"{config['test']['output_dir']}MF0012_src_{config['test']['set']}_{config['main']['trg_facility']}.txt",
        format="%(levelname)s: %(message)s",
    )

    for cv_num in range(config["main"]["cv"]):
        logging.info(f"\n\n\n== CV{cv_num} ==")
        weight_path = weight_list[cv_num]
        # project_prefix = f"cv{cv_num}_{config['main']['src_facility']}_src_"
        project_prefix = f"cv{cv_num}_src-only_{config['main']['src_facility']}_"

        project = (
            project_prefix
            + config["main"]["model"]
            + "_"
            + config["main"]["optim"]
            + "_batch"
            + str(config["main"]["batch_size"])
            + "_shape"
            + str(config["main"]["shape"])
        )
        logging.info(f"{project}\n")

        # # --- sourceデータ --- #
        # wsis = joblib.load(
        #     config["main"]["jb_dir"]
        #     + f"{config['main']['src_facility']}/"
        #     + f"cv{cv_num}_"
        #     + f"{config['test']['set']}_"
        #     + f"{config['main']['trg_facility']}_wsi.jb"
        # )

        # --- tragetデータ --- #
        wsis = joblib.load(
            config["main"]["jb_dir"]
            + f"{config['main']['trg_facility']}/"
            + "cv0_train_"
            + f"{config['main']['trg_facility']}_wsi.jb"
        )
        wsis += joblib.load(
            config["main"]["jb_dir"]
            + f"{config['main']['trg_facility']}/"
            + "cv0_valid_"
            + f"{config['main']['trg_facility']}_wsi.jb"
        )
        wsis += joblib.load(
            config["main"]["jb_dir"]
            + f"{config['main']['trg_facility']}/"
            + "cv0_test_"
            + f"{config['main']['trg_facility']}_wsi.jb"
        )

        # wsisから訓練に使用されたtargetのWSIを除去（比較対象を含む）
        print(f"[WSI num] before remove: {len(wsis)}")
        for selected_wsi in config["main"]["trg_selected_wsis"]:
            wsis.remove(selected_wsi)
        print(f"[WSI num] after remove (trg_selected_wsis): {len(wsis)}")
        for valid_wsi in config["main"]["trg_valid_wsis"]:
            wsis.remove(valid_wsi)
        print(f"[WSI num] after remove (valid_wsis): {len(wsis)}")
        # ------------- #

        files = get_files(
            wsis, config["main"]["classes"], config["dataset"]["trg_imgs_dir"]
        )

        if len(config["main"]["classes"]) > 1:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.BCEWithLogitsLoss()

        net = build_model(
            config["main"]["model"], num_classes=len(config["main"]["classes"])
        )

        logging.info("Loading model {}".format(weight_path))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device {device}")
        net.to(device=device)
        net.load_state_dict(torch.load(weight_path, map_location=device))

        dataset = WSI(
            files,
            config["main"]["classes"],
            tuple(config["main"]["shape"]),
            transform={"Resize": True, "HFlip": False, "VFlip": False},
        )

        loader = DataLoader(
            dataset,
            batch_size=config["main"]["batch_size"],
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        val_loss, cm = eval_net_test(
            net,
            loader,
            criterion,
            device,
            get_miss=config["test"]["get_miss"],
            save_dir=config["test"]["output_dir"],
        )

        logging.info(
            f"\n cm ({config['test']['set']}):\n{np.array2string(cm, separator=',')}\n"
        )
        val_metrics = eval_metrics(cm)
        logging.info("===== eval metrics =====")
        logging.info(
            f"\n Accuracy ({config['test']['set']}):  {val_metrics['accuracy']}"
        )
        logging.info(
            f"\n Precision ({config['test']['set']}): {val_metrics['precision']}"
        )
        logging.info(f"\n Recall ({config['test']['set']}):    {val_metrics['recall']}")
        logging.info(f"\n F1 ({config['test']['set']}):        {val_metrics['f1']}")
        logging.info(f"\n mIoU ({config['test']['set']}):      {val_metrics['mIoU']}")

        # Not-Normalized
        cm_plt = plot_confusion_matrix(cm, config["main"]["classes"], normalize=False)
        cm_plt.savefig(
            config["test"]["output_dir"]
            + project
            + f"_{config['test']['set']}_{config['main']['trg_facility']}_nn-confmatrix.png"
        )
        plt.clf()
        plt.close()

        # Normalized
        cm_plt = plot_confusion_matrix(cm, config["main"]["classes"], normalize=True)
        cm_plt.savefig(
            config["test"]["output_dir"]
            + project
            + f"_{config['test']['set']}_{config['main']['trg_facility']}_confmatrix.png"
        )
        plt.clf()
        plt.close()


def main_trg():
    """
    source+1targetで訓練されたモデルのテスト
    """
    fix_seed(0)
    config_path = "../config/config_t1_cl[0, 1, 2]_val-t3.yaml"
    # config_path = "../config/config_st1_cl[0, 1, 2]_val-t3.yaml"

    with open(config_path) as file:
        config = yaml.safe_load(file.read())

    logging.basicConfig(
        level=logging.INFO,
        filename=f"{config['test']['output_dir']}{config['main']['prefix']}_{config['test']['set']}_{config['main']['trg_facility']}.txt",
        format="%(levelname)s: %(message)s",
    )

    for cv_num in range(config["main"]["cv"]):
        for trg_selected_wsi in config["main"]["trg_selected_wsis"]:
            logging.info(f"\n\n\n== CV{cv_num}: {trg_selected_wsi} ==")

            # 学習済み重みの読み込み
            weight_dir = f"{config['test']['weight_dir']}{config['main']['prefix']}_MF0012_{trg_selected_wsi}_{config['main']['classes']}/"
            # cv_numとtrg_selected_wsiがマッチするweight_pathをリストで取得
            weight_names = natsorted(os.listdir(weight_dir))
            weight_paths = [
                weight_dir + s
                for s in weight_names
                if re.match(f"cv{cv_num}_.*{trg_selected_wsi}_.*", s)
            ]
            print(weight_paths)

            if len(weight_paths) == 1:
                weight_path = weight_paths[0]
            else:
                sys.exit(f"No weight path exists for cv{cv_num} & {trg_selected_wsi}")

            # project_prefix = (
            #     f"cv{cv_num}_{config['main']['src_facility']}_{trg_selected_wsi}_"
            # )
            project_prefix = (
                f"cv{cv_num}_{config['main']['prefix']}_{config['main']['src_facility']}_{trg_selected_wsi}_"
            )

            project = (
                project_prefix
                + config["main"]["model"]
                + "_"
                + config["main"]["optim"]
                + "_batch"
                + str(config["main"]["batch_size"])
                + "_shape"
                + str(config["main"]["shape"])
            )
            logging.info(f"{project}\n")

            # --- traget用 --- #
            wsis = joblib.load(
                config["main"]["jb_dir"]
                + f"{config['main']['trg_facility']}/"
                + "cv0_train_"
                + f"{config['main']['trg_facility']}_wsi.jb"
            )
            wsis += joblib.load(
                config["main"]["jb_dir"]
                + f"{config['main']['trg_facility']}/"
                + "cv0_valid_"
                + f"{config['main']['trg_facility']}_wsi.jb"
            )
            wsis += joblib.load(
                config["main"]["jb_dir"]
                + f"{config['main']['trg_facility']}/"
                + "cv0_test_"
                + f"{config['main']['trg_facility']}_wsi.jb"
            )

            # wsisから訓練に使用されたtargetのWSIを除去（比較対象を含む）
            print(f"[WSI num] before remove: {len(wsis)}")
            for selected_wsi in config["main"]["trg_selected_wsis"]:
                wsis.remove(selected_wsi)
            print(f"[WSI num] after remove (trg_selected_wsis): {len(wsis)}")
            for valid_wsi in config["main"]["trg_valid_wsis"]:
                wsis.remove(valid_wsi)
            print(f"[WSI num] after remove (valid_wsis): {len(wsis)}")
            # ------------- #

            files = get_files(
                wsis, config["main"]["classes"], config["dataset"]["trg_imgs_dir"]
            )

            if len(config["main"]["classes"]) > 1:
                criterion = nn.CrossEntropyLoss()
            else:
                criterion = nn.BCEWithLogitsLoss()

            net = build_model(
                config["main"]["model"], num_classes=len(config["main"]["classes"])
            )

            logging.info("Loading model {}".format(weight_path))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logging.info(f"Using device {device}")
            net.to(device=device)
            net.load_state_dict(torch.load(weight_path, map_location=device))

            dataset = WSI(
                files,
                config["main"]["classes"],
                tuple(config["main"]["shape"]),
                transform={"Resize": True, "HFlip": False, "VFlip": False},
            )

            loader = DataLoader(
                dataset,
                batch_size=config["main"]["batch_size"],
                shuffle=False,
                num_workers=2,
                pin_memory=True,
            )
            val_loss, cm = eval_net_test(
                net,
                loader,
                criterion,
                device,
                get_miss=config["test"]["get_miss"],
                save_dir=config["test"]["output_dir"],
            )

            logging.info(
                f"\n cm ({config['test']['set']}):\n{np.array2string(cm, separator=',')}\n"
            )
            val_metrics = eval_metrics(cm)
            logging.info("===== eval metrics =====")
            logging.info(
                f"\n Accuracy ({config['test']['set']}):  {val_metrics['accuracy']}"
            )
            logging.info(
                f"\n Precision ({config['test']['set']}): {val_metrics['precision']}"
            )
            logging.info(
                f"\n Recall ({config['test']['set']}):    {val_metrics['recall']}"
            )
            logging.info(f"\n F1 ({config['test']['set']}):        {val_metrics['f1']}")
            logging.info(
                f"\n mIoU ({config['test']['set']}):      {val_metrics['mIoU']}"
            )

            # Not-Normalized
            cm_plt = plot_confusion_matrix(
                cm, config["main"]["classes"], normalize=False
            )
            cm_plt.savefig(
                config["test"]["output_dir"]
                + project
                + f"_{config['test']['set']}_{config['main']['trg_facility']}_nn-confmatrix.png"
            )
            plt.clf()
            plt.close()

            # Normalized
            cm_plt = plot_confusion_matrix(
                cm, config["main"]["classes"], normalize=True
            )
            cm_plt.savefig(
                config["test"]["output_dir"]
                + project
                + f"_{config['test']['set']}_{config['main']['trg_facility']}_confmatrix.png"
            )
            plt.clf()
            plt.close()


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    main_trg()
    # main_src()
