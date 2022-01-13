import os
import sys
import torch
import joblib
import yaml
import logging
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from S.util import fix_seed
from visualyze2.feature import get_latent_vecs_list
from visualyze2.plot import plot_feature_space
from visualyze2.fe_model import netE_midlayer


def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description="Feature space")
    parser.add_argument(
        "--title",
        type=str,
        default="st1-ada_valt3_MF0012_cl[0, 1, 2]",
        metavar="N",
        help="title of this project",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/secssd/AL_SSDA_WSI_strage/st_ada_result/featuremap/",
        metavar="N",
        help="dir of visualizing results",
    )
    parser.add_argument(
        "--cl_labels",
        type=list,
        default=["Non-Neop.", "HSIL", "LSIL"],
        metavar="N",
        help="domain name list",
    )
    parser.add_argument(
        "--sample-N",
        type=int,
        default=500,
        metavar="N",
        help="Each class-sample num",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    args = parser.parse_args()
    return args


# For multiple domains
def main(config_path: str, ):
    fix_seed(0)

    args = get_args()
    with open(config_path) as file:
        config = yaml.safe_load(file.read())

    input_shape = tuple(config['main']['shape'])

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    # load wsi_list
    logging.info("load wsi list...")
    # WSIのリストを取得 (target)
    trg_l_train_wsis = joblib.load(
        config['dataset']['jb_dir']
        + f"{config['main']['trg_facility']}/"
        + "trg_l_wsi.jb"
    )
    # trg_valid_wsis = joblib.load(
    #     config['dataset']['jb_dir']
    #     + f"{config['main']['trg_facility']}/"
    #     + "valid_wsi.jb"
    # )
    trg_unl_test_wsis = joblib.load(
        config['dataset']['jb_dir']
        + f"{config['main']['trg_facility']}/"
        + "trg_unl_wsi.jb"
    )

    trg_l_wsis = trg_l_train_wsis
    trg_unl_wsis = trg_unl_test_wsis

    for cv_num in range(config["main"]["cv"]):
        for trg_l_num, trg_l_selected_wsi in enumerate(trg_l_wsis):
            if trg_l_num != 0:
                continue

            suffix = config['test']['weight_names'][trg_l_selected_wsi][cv_num]
            suffix = suffix.replace(f"cv{cv_num}_E_", "").replace(".pth", "")

            # title = f"{args.title}_cv{cv_num}_{trg_l_selected_wsi}"
            title = f"{args.title}_{suffix}_cv{cv_num}_{trg_l_selected_wsi}"
            logging.info(f"== CV{cv_num}: {trg_l_selected_wsi} ==")

            # WSIのリストを取得 (source)
            src_l_train_wsis = joblib.load(
                config['dataset']['jb_dir']
                + f"{config['main']['src_facility']}/"
                + f"cv{cv_num}_"
                + f"train_{config['main']['src_facility']}_wsi.jb"
            )
            src_l_wsis = src_l_train_wsis

            # load model
            logging.info("set model...")
            weight_path = (
                config['test']['weight_dir'][trg_l_selected_wsi]
                + config['test']['weight_names'][trg_l_selected_wsi][cv_num]
            )

            model = netE_midlayer(
                encoder_name=config['main']['model'],
                num_classes=len(config['main']['classes']),
                weight_path=weight_path,
            ).to(device=device)

            # visualize each domain's feature space
            src_l_vecs_list = get_latent_vecs_list(
                model, wsi_list=src_l_wsis,
                imgs_dir=config['dataset']['src_imgs_dir'],
                sample_N=args.sample_N,
                classes=config['main']['classes'],
                input_shape=input_shape,
                batch_size=args.batch_size,
                output_dir=None
            )
            trg_l_vecs_list = get_latent_vecs_list(
                model, wsi_list=trg_l_wsis,
                imgs_dir=config['dataset']['trg_imgs_dir'],
                sample_N=args.sample_N,
                classes=config['main']['classes'],
                input_shape=input_shape,
                batch_size=args.batch_size,
                output_dir=None
            )
            trg_unl_vecs_list = get_latent_vecs_list(
                model, wsi_list=trg_unl_wsis,
                imgs_dir=config['dataset']['trg_imgs_dir'],
                sample_N=args.sample_N,
                classes=config['main']['classes'],
                input_shape=input_shape,
                batch_size=args.batch_size,
                output_dir=None
            )

            # trg_l_vecs_list = [[], [], []]
            # trg_unl_vecs_list = [[], [], []]
            # prototype_vecs_list = [[], [], []]

            logging.info("plot feature space")
            # pca
            plot_feature_space(
                src_l_vecs_list=src_l_vecs_list,
                trg_l_vecs_list=trg_l_vecs_list,
                trg_unl_vecs_list=trg_unl_vecs_list,
                method="pca",
                output_dir=args.output_dir,
                title=title
            )

            # tsne
            plot_feature_space(
                src_l_vecs_list=src_l_vecs_list,
                trg_l_vecs_list=trg_l_vecs_list,
                trg_unl_vecs_list=trg_unl_vecs_list,
                method="tsne",
                output_dir=args.output_dir,
                title=title
            )

            # umap
            plot_feature_space(
                src_l_vecs_list=src_l_vecs_list,
                trg_l_vecs_list=trg_l_vecs_list,
                trg_unl_vecs_list=trg_unl_vecs_list,
                method="umap",
                output_dir=args.output_dir,
                title=title
            )


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    config_path = "../ST_ADA/config_st-ada_cl[0, 1, 2]_valt3.yaml"
    # config_path = "./ST_ADA/config_st-ada_cl[0, 1, 2]_valt3.yaml"
    main(config_path=config_path)
