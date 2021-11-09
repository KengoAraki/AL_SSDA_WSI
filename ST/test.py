import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from S.test import main_trg, main_src


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    config_path = "../ST/config_st_cl[0, 1, 2]_valt3.yaml"
    # config_path = "../ST/config_st_cl[0, 1, 2]_valt3._pretrained.yaml"

    main_trg(config_path=config_path, test_set="trg_unl")
    main_trg(config_path=config_path, test_set="trg_l")
    main_trg(config_path=config_path, test_set="valid")

    # main_src(config_path=config_path, test_set="train")
