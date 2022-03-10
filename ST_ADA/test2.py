import os
import sys
import logging
import yaml
import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from S.eval import eval_metrics, plot_confusion_matrix
from S.dataset import WSI, get_files
from S.util import fix_seed
from ST_ADA.model import Encoder
from ST_ADA.eval import eval_net_test


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
    rotation: int = 0,
):
    criterion = nn.CrossEntropyLoss()

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
    _, cm = eval_net_test(
        net,
        loader,
        criterion,
        device,
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
    logging.info(f"\n Dice ({test_set}):      {val_metrics['dice']}")
    logging.info(f"\n mIoU ({test_set}):      {val_metrics['mIoU']}")

    # 軸入れ替え
    cm_rep = np.copy(cm)
    cm_rep = cm_rep[:, [0, 2, 1]]
    cm_rep = cm_rep[[0, 2, 1], :]
    cl_labels = ["Non-\nNeop.", "LSIL", "HSIL"]

    # Not-Normalized
    cm_plt = plot_confusion_matrix(cm_rep, cl_labels, normalize=False, font_size=25, rotation=rotation)
    cm_plt.savefig(
        output_dir
        + project
        + "_nn-confmatrix.png"
    )
    plt.clf()
    plt.close()

    # Normalized
    cm_plt = plot_confusion_matrix(cm_rep, cl_labels, normalize=True, font_size=35, rotation=rotation)
    cm_plt.savefig(
        output_dir
        + project
        + "_confmatrix.png"
    )
    plt.clf()
    plt.close()

    return cm


def main_trg(trg_l_wsi: str, config_path: str, test_set: str = "trg_unl"):
    fix_seed(0)
    rotation = 0

    # ==== load config ===== #
    with open(config_path) as file:
        config = yaml.safe_load(file.read())

    output_dir = (
        f"{config['test']['output_dir']}"
        + f"{config['main']['prefix']}_{config['main']['src_facility']}_{trg_l_wsi}_{config['main']['classes']}/")
    if os.path.exists(output_dir) is False:
        os.mkdir(output_dir)

    weight_list = [
        config['test']['weight_dir'][trg_l_wsi] + name
        for name
        in config['test']['weight_names'][trg_l_wsi]
    ]

    logging.basicConfig(
        level=logging.INFO,
        filename=f"{output_dir}{config['main']['prefix']}_{config['main']['src_facility']}_{trg_l_wsi}_{test_set}-{config['main']['trg_facility']}.txt",
        format="%(levelname)s: %(message)s",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for cv_num in range(config['main']['cv']):

        logging.info(f"\n\n\n== CV{cv_num} ==")
        weight_path = weight_list[cv_num]
        project_prefix = f"{config['main']['prefix']}_{config['main']['src_facility']}_{trg_l_wsi}_cv{cv_num}_"

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
            + config['main']['trg_facility']
        )
        logging.info(f"{project}\n")

        # --- targetデータ --- #
        wsis = joblib.load(
            config['dataset']['jb_dir']
            + f"{config['main']['trg_facility']}/"
            + f"{test_set}_wsi.jb"
        )
        files = get_files(
            wsis, config['main']['classes'], config['dataset']['trg_imgs_dir']
        )
        # ------------- #

        net = Encoder(
            encoder_name=config['main']['model'],
            num_classes=len(config['main']['classes']),
            pretrained=False, weight_path=None, device=device
        ).to(device)
        logging.info("Loading model {}".format(weight_path))
        net.load_state_dict(torch.load(weight_path, map_location=device))

        cm = test_net(
            net=net,
            files=files,
            classes=config['main']['classes'],
            test_set=test_set,
            output_dir=output_dir,
            project=project,
            device=device,
            shape=tuple(config['main']['shape']),
            batch_size=config['main']['batch_size'],
            rotation=rotation
        )

        if cv_num == 0:
            cm_all = cm
        else:
            cm_all += cm

    # ===== cv_all ===== #
    logging.info("\n\n== ALL ==")
    project_prefix = f"{config['main']['prefix']}_{config['main']['src_facility']}_{trg_l_wsi}_all_"

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
        + config['main']['trg_facility']
    )
    logging.info(f"{project}\n")

    logging.info(
        f"\n cm ({test_set}):\n{np.array2string(cm_all, separator=',')}\n"
    )
    val_metrics = eval_metrics(cm_all)
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

    # 軸入れ替え
    cm_all = cm_all[:, [0, 2, 1]]
    cm_all = cm_all[[0, 2, 1], :]
    cl_labels = ["Non-\nNeop.", "LSIL", "HSIL"]

    # Not-Normalized
    cm_plt = plot_confusion_matrix(cm_all, cl_labels, normalize=False, font_size=25, rotation=rotation)
    cm_plt.savefig(
        output_dir
        + project
        + "_nn-confmatrix.png"
    )
    plt.clf()
    plt.close()

    # Normalized
    cm_plt = plot_confusion_matrix(cm_all, cl_labels, normalize=True, font_size=35, rotation=rotation)
    cm_plt.savefig(
        output_dir
        + project
        + "_confmatrix.png"
    )
    plt.clf()
    plt.close()


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    # config_path = "./ST_ADA/config_st-ada_cl[0, 1, 2]_valt3.yaml"
    config_path = "../ST_ADA/config_st-ada_cl[0, 1, 2]_valt3.yaml"

    # config_path = "../ST_ADA/config_st-ada_cl[0, 1, 2]_valt3_unl-trg-balance.yaml"
    # main_trg(trg_l_wsi='03_G144', config_path=config_path, test_set="trg_unl")

    config_path = "../ST_ADA/config_st-ada_cl[0, 1, 2]_valt3.yaml"
    main_trg(trg_l_wsi='03_G144', config_path=config_path, test_set="trg_unl")
    config_path = "../ST_ADA/config_st-ada_cl[0, 1, 2]_valt3_unl-trg-balance.yaml"
    main_trg(trg_l_wsi='03_G144', config_path=config_path, test_set="trg_unl")
    config_path = "../ST_ADA/config_st-ada_cl[0, 1, 2]_valt3_clf-imbalance.yaml"
    main_trg(trg_l_wsi='03_G144', config_path=config_path, test_set="trg_unl")
