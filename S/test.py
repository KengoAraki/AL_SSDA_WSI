import logging
import os
import sys
import yaml
import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from S.dataset import WSI, get_files
from S.eval import eval_net_test, plot_confusion_matrix, eval_metrics
from S.util import fix_seed
from S.model import build_model


def test_net(
    net,
    files: list,
    classes: list,
    test_set: str,
    output_dir: str,
    project: str = "test_net",
    device=torch.device('cuda'),
    shape: tuple = (256, 256),
    batch_size: int = 32,
    get_miss: bool = False,
):
    if len(classes) > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    dataset = WSI(
        files,
        classes,
        shape,
        transform={"Resize": True, "HFlip": False, "VFlip": False},
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    val_loss, cm = eval_net_test(
        net,
        loader,
        criterion,
        device,
        get_miss=get_miss,
        save_dir=output_dir,
    )

    logging.info(
        f"\n cm ({test_set}):\n{np.array2string(cm, separator=',')}\n"
    )
    val_metrics = eval_metrics(cm)
    logging.info("===== eval metrics =====")
    logging.info(
        f"\n Accuracy ({test_set}):  {val_metrics['accuracy']}"
    )
    logging.info(
        f"\n Precision ({test_set}): {val_metrics['precision']}"
    )
    logging.info(f"\n Recall ({test_set}):    {val_metrics['recall']}")
    logging.info(f"\n F1 ({test_set}):        {val_metrics['f1']}")
    logging.info(f"\n mIoU ({test_set}):      {val_metrics['mIoU']}")

    # Not-Normalized
    cm_plt = plot_confusion_matrix(cm, classes, normalize=False)
    cm_plt.savefig(
        output_dir
        + project
        + "_nn-confmatrix.png"
    )
    plt.clf()
    plt.close()

    # Normalized
    cm_plt = plot_confusion_matrix(cm, classes, normalize=True)
    cm_plt.savefig(
        output_dir
        + project
        + "_confmatrix.png"
    )
    plt.clf()
    plt.close()


def main_src(config_path: str, test_set: str = "test"):
    """
    sourceのみで訓練されたモデルを使用
    source dataに対してテスト
    """
    fix_seed(0)

    with open(config_path) as file:
        config = yaml.safe_load(file.read())

    weight_list = [
        config['test']['weight_dir'] + name for name in config['test']['weight_names']
    ]

    logging.basicConfig(
        level=logging.INFO,
        filename=f"{config['test']['output_dir']}s_{config['main']['facility']}_{test_set}-{config['test']['src_facility']}.txt",
        format="%(levelname)s: %(message)s",
    )

    for cv_num in range(config['main']['cv']):
        logging.info(f"\n\n\n== CV{cv_num} ==")
        weight_path = weight_list[cv_num]
        project_prefix = f"s_{config['main']['facility']}_cv{cv_num}_"

        project = (
            project_prefix
            + config['main']['model']
            + "_"
            + config['main']['optim']
            + "_batch"
            + str(config['main']['batch_size'])
            + "_shape"
            + str(config['main']['shape'])
            + "_"
            + test_set
            + "-"
            + config['test']['src_facility']
        )
        logging.info(f"{project}\n")

        # --- sourceデータ --- #
        wsis = joblib.load(
            config['dataset']['jb_dir']
            + f"{config['test']['src_facility']}/"
            + f"cv{cv_num}_"
            + f"{test_set}_"
            + f"{config['test']['src_facility']}_wsi.jb"
        )
        files = get_files(
            wsis, config['main']['classes'], config['test']['src_imgs_dir']
        )
        # ------------- #

        net = build_model(
            config['main']['model'], num_classes=len(config['main']['classes'])
        )
        logging.info("Loading model {}".format(weight_path))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net.to(device=device)
        net.load_state_dict(torch.load(weight_path, map_location=device))

        test_net(
            net=net,
            files=files,
            classes=config['main']['classes'],
            test_set=test_set,
            output_dir=config['test']['output_dir'],
            project=project,
            device=device,
            shape=tuple(config['main']['shape']),
            batch_size=config['main']['batch_size'],
        )


def main_trg(config_path: str, test_set: str = "trg_unl"):
    """
    sourceのみで訓練されたモデルを使用
    target dataに対してテスト
    """
    fix_seed(0)

    with open(config_path) as file:
        config = yaml.safe_load(file.read())

    weight_list = [
        config['test']['weight_dir'] + name for name in config['test']['weight_names']
    ]

    logging.basicConfig(
        level=logging.INFO,
        filename=f"{config['test']['output_dir']}s_{config['main']['facility']}_{test_set}-{config['test']['trg_facility']}.txt",
        format="%(levelname)s: %(message)s",
    )

    for cv_num in range(config['main']['cv']):
        logging.info(f"\n\n\n== CV{cv_num} ==")
        weight_path = weight_list[cv_num]
        project_prefix = f"s_{config['main']['facility']}_cv{cv_num}_"

        project = (
            project_prefix
            + config['main']['model']
            + "_"
            + config['main']['optim']
            + "_batch"
            + str(config['main']['batch_size'])
            + "_shape"
            + str(config['main']['shape'])
            + "_"
            + test_set
            + "-"
            + config['test']['trg_facility']
        )
        logging.info(f"{project}\n")

        # --- targetデータ --- #
        wsis = joblib.load(
            config['dataset']['jb_dir']
            + f"{config['test']['trg_facility']}/"
            + f"{test_set}_wsi.jb"
        )
        files = get_files(
            wsis, config['main']['classes'], config['test']['trg_imgs_dir']
        )
        # ------------- #

        net = build_model(
            config['main']['model'], num_classes=len(config['main']['classes'])
        )
        logging.info("Loading model {}".format(weight_path))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net.to(device=device)
        net.load_state_dict(torch.load(weight_path, map_location=device))

        test_net(
            net=net,
            files=files,
            classes=config['main']['classes'],
            test_set=test_set,
            output_dir=config['test']['output_dir'],
            project=project,
            device=device,
            shape=tuple(config['main']['shape']),
            batch_size=config['main']['batch_size'],
        )


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    config_path = "../S/config_s_cl[0, 1, 2].yaml"
    # config_path = "./S/config_s_cl[0, 1, 2].yaml"  # For debug

    main_trg(config_path=config_path, test_set="trg_unl")
    main_src(config_path=config_path, test_set="test")
