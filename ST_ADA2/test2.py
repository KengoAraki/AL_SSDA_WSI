import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from ST_ADA.test2 import main_trg

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    # config_path = "./ST_ADA2/config_st-ada2_cl[0, 1, 2]_valt3.yaml"
    config_path = "../ST_ADA2/config_st-ada2_cl[0, 1, 2]_valt3.yaml"

    main_trg(trg_l_wsi='03_G144', config_path=config_path, test_set="trg_unl")
