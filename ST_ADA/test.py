import os
import sys
import logging
import yaml
import joblib
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dataset import Source_Dataset, Target_Dataset
from eval import plot_confusion_matrix, eval_metrics, get_confusion_matrix
from util import fix_seed
from model import Encoder

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# FIXME: 要修正

def test_net(
    netE,
    loader,
    device,
    batch,
    classes,
    output_dir,
    project,
    data_mode="test"
):
    netE.eval()

    n_val = len(loader)  # the number of batch
    init_flag = True

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for i, data in enumerate(loader):
            img = data["img"].to(device)
            gt = data["gt"].to(device)

            with torch.no_grad():
                pred = netE(img, mode="class")

            # confusion matrix
            if netE.fc.out_features > 1:
                pred = nn.Softmax(dim=1)(pred)
            if init_flag:
                cm = get_confusion_matrix(pred, gt)
                init_flag = False
            else:
                cm += get_confusion_matrix(pred, gt)

            pbar.update()

    logging.info(f"\n cm ({data_mode}):\n{cm}\n")
    val_metrics = eval_metrics(cm)
    logging.info(f"\n Accuracy ({data_mode}):  {val_metrics['accuracy']}")
    logging.info(f"\n Precision ({data_mode}): {val_metrics['precision']}")
    logging.info(f"\n Recall ({data_mode}):    {val_metrics['recall']}")

    # Not-Normalized
    cm_plt = plot_confusion_matrix(
        cm, classes, normalize=False)
    cm_plt.savefig(
        output_dir
        + project
        + "_nn-confmatrix.png"
    )
    plt.clf()
    plt.close()

    # Normalized
    cm_plt = plot_confusion_matrix(
        cm, classes, normalize=True)
    cm_plt.savefig(
        output_dir
        + project
        + "_confmatrix.png"
    )
    plt.clf()
    plt.close()


def test():
    fix_seed(0)

    # ==== load config ===== #
    # config_path = "./config/config.yaml"
    config_path = "../config/config.yaml"
    with open(config_path) as file:
        config = yaml.safe_load(file.read())

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    classes = config['main']['classes']
    batch = config['main']['batch']
    encoder_name = config['main']['encoder_name']
    shape = tuple(config['main']['shape'])
    cv_num = config['main']['cv_num']
    num_classes = len(classes)

    dataset_dir = config['dataset']['dataset_dir']
    jb_dir = config['dataset']['jb_dir']
    source = config['dataset']['source']
    target = config['dataset']['target']

    output_dir = config['test']['output_dir']
    data_mode = config['test']['data_mode']
    weight_path = config['test']['weight_path']

    transform = {'Resize': True, 'HFlip': False, 'VFlip': False}
    num_workers = 2

    # project = f"ADA_src-{source}_trg-{target}_{encoder_name}_cl{classes}_{data_mode}_cv{cv_num}"
    project = f"cv{cv_num}_ADA_src-{source}_trg-{target}_{encoder_name}_cl{classes}_{data_mode}"
    logging.basicConfig(
        level=logging.INFO,
        filename=f"{output_dir}{project}.txt",
        format='%(levelname)s: %(message)s'
    )
    logging.info(f"{project}\n")

    # load model
    logging.info("Loading model {}".format(weight_path))
    netE = Encoder(encoder_name, num_classes, pretrained=False).to(device)
    netE.load_state_dict(
        torch.load(weight_path, map_location=device))
    logging.info("Model loaded !")

    # === Source data === #
    logging.info("=== test source data ===")
    test_source_wsi = joblib.load(
        jb_dir
        + f"source_{source}/"
        + f"cv{cv_num}_{data_mode}_source-{source}_wsi.jb"
    )

    test_source_dataset = Source_Dataset(
        test_source_wsi,
        imgs_dir=f"{dataset_dir}{source}/",
        classes=classes,
        shape=shape,
        transform=transform,
        mode="test"
    )
    test_source_loader = DataLoader(
        test_source_dataset, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    num_test_source = test_source_dataset.__len__()
    logging.info(f"num_test_source: {num_test_source}")

    test_net(
        netE,
        test_source_loader,
        device,
        batch,
        classes,
        output_dir,
        f"{project}_source",
        data_mode=data_mode
    )

    # === Target data === #
    logging.info("=== test target data ===")
    test_target_wsi = joblib.load(
        jb_dir
        + f"target_{target}/"
        + f"cv{cv_num}_{data_mode}_target-{target}_wsi.jb"
    )
    if data_mode == "train":
        test_target_wsi += joblib.load(
            jb_dir
            + f"target_{target}/"
            + f"cv{cv_num}_valid_target-{target}_wsi.jb"
        )

    test_target_dataset = Target_Dataset(
        test_target_wsi,
        imgs_dir=f"{dataset_dir}{target}/",
        classes=classes,
        shape=shape,
        transform=transform,
        mode="test"
    )
    test_target_loader = DataLoader(
        test_target_dataset, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    num_test_target = test_target_dataset.__len__()
    logging.info(f"num_test_target: {num_test_target}")

    test_net(
        netE,
        test_target_loader,
        device,
        batch,
        classes,
        output_dir,
        f"{project}_target",
        data_mode=data_mode
    )


if __name__ == "__main__":
    test()
