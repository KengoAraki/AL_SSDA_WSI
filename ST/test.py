import os
import sys
import logging
import yaml
import joblib
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from S.dataset import get_files
from S.util import fix_seed
from S.model import build_model
from S.test import test_net


def main_trg(trg_l_wsi: str, config_path: str, test_set: str = "trg_unl"):
    """
    source + target 1枚のみで訓練されたモデルを使用
    target dataに対してテスト
    """
    fix_seed(0)

    with open(config_path) as file:
        config = yaml.safe_load(file.read())

    weight_list = [
        config['test']['weight_dir'][trg_l_wsi] + name
        for name
        in config['test']['weight_names'][trg_l_wsi]
    ]

    logging.basicConfig(
        level=logging.INFO,
        filename=f"{config['test']['output_dir']}st1_{config['main']['src_facility']}_{trg_l_wsi}_{test_set}-{config['main']['trg_facility']}.txt",
        format="%(levelname)s: %(message)s",
    )

    for cv_num in range(config['main']['cv']):

        logging.info(f"\n\n\n== CV{cv_num} ==")
        weight_path = weight_list[cv_num]
        project_prefix = f"st1_{config['main']['src_facility']}_{trg_l_wsi}_cv{cv_num}_"

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
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    # config_path = "../ST/config_st_cl[0, 1, 2]_valt3.yaml"
    config_path = "../ST/config_st_cl[0, 1, 2]_valt3_pretrained.yaml"

    main_trg(trg_l_wsi='03_G144', config_path=config_path, test_set="trg_unl")
    main_trg(trg_l_wsi='03_G293', config_path=config_path, test_set="trg_unl")
    main_trg(trg_l_wsi='03_G109-1', config_path=config_path, test_set="trg_unl")
