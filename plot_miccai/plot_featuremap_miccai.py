import os
import sys
import torch
import joblib
import yaml
import logging
import argparse
import copy
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from S.util import fix_seed
from plot_miccai.feature import get_latent_vecs_list, get_all_class_files_list
from plot_miccai.plot import plot_feature_space_miccai
from visualyze2.fe_model import resnet50_midlayer


def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description="Feature space")
    parser.add_argument(
        "--title",
        type=str,
        default="st1_valt20_srcMF0012_trgMF0003_cl[0, 1, 2]_best_mIoU",
        metavar="N",
        help="title of this project",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/secssd/AL_SSDA_WSI_MICCAI_strage/st_pretrained_result/featuremap/",
        metavar="N",
        help="dir of visualizing results",
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
def main(
    config_path: str,
    l_trg_wsi: str = '03_G144',
    l_trg_set: str = 'top',
    cv_num: int = 0,
    is_pretrained_model: bool = False,
):
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
        + "trg_l_top_wsi.jb"
    )
    trg_l_train_wsis += joblib.load(
        config['dataset']['jb_dir']
        + f"{config['main']['trg_facility']}/"
        + "trg_l_med_wsi.jb"
    )
    trg_l_train_wsis += joblib.load(
        config['dataset']['jb_dir']
        + f"{config['main']['trg_facility']}/"
        + "trg_l_btm_wsi.jb"
    )
    trg_unl_test_wsis = joblib.load(
        config['dataset']['jb_dir']
        + f"{config['main']['trg_facility']}/"
        + "trg_unl_wsi.jb"
    )

    other_wsis = copy.deepcopy(trg_l_train_wsis)
    if is_pretrained_model is False:
        other_wsis.remove(l_trg_wsi)

    trg_unl_wsis = trg_unl_test_wsis

    if is_pretrained_model:
        title = f"{args.title}_cv{cv_num}_{l_trg_wsi}_init"
    else:
        title = f"{args.title}_cv{cv_num}_{l_trg_wsi}"
    logging.info(f"== CV{cv_num}: {l_trg_wsi} ==")

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

    # for pretrained model
    if is_pretrained_model:
        weight_path = (
            config['main']['pretrained_weight_dir']
            + config['main']['pretrained_weight_names'][cv_num]
        )
    else:  # for test model
        weight_list = [
            f"{config['test']['weight_dir']}{config['main']['prefix']}_{config['main']['src_facility']}_{l_trg_set}_{l_trg_wsi}_{config['main']['classes']}/" + name
            for name
            in config['test']['weight_names'][l_trg_set][l_trg_wsi]
        ]
        weight_path = weight_list[cv_num]

    model = resnet50_midlayer(
        num_classes=len(config['main']['classes']),
        weight_path=weight_path,
    ).to(device=device)

    # # 各クラスのfile listのlistを取得
    # src_l_all_class_files_list = get_all_class_files_list(
    #     wsi_list=src_l_wsis,
    #     imgs_dir=config['dataset']['src_imgs_dir'],
    #     classes=config['main']['classes']
    # )
    trg_l_all_class_files_list = get_all_class_files_list(
        wsi_list=[l_trg_wsi],
        imgs_dir=config['dataset']['trg_imgs_dir'],
        classes=config['main']['classes']
    )
    # trg_unl_all_class_files_list = get_all_class_files_list(
    #     wsi_list=trg_unl_wsis,
    #     imgs_dir=config['dataset']['trg_imgs_dir'],
    #     classes=config['main']['classes']
    # )
    other_all_class_files_list = get_all_class_files_list(
        wsi_list=other_wsis,
        imgs_dir=config['dataset']['trg_imgs_dir'],  # 要チェック
        classes=config['main']['classes']
    )

    # # 最初だけ!
    # joblib.dump(
    #     (src_l_all_class_files_list, trg_l_all_class_files_list, trg_unl_all_class_files_list, other_all_class_files_list),
    #     f"/home/kengoaraki/Project/DA/AL_SSDA_WSI/plot_miccai/output/all_class_files_list_{l_trg_set}_{l_trg_wsi}.jb",
    #     compress=3
    # )
    src_l_all_class_files_list, _, trg_unl_all_class_files_list, _ = \
        joblib.load("/home/kengoaraki/Project/DA/AL_SSDA_WSI/plot_miccai/output/all_class_files_list_top_03_G144.jb")

    # # visualize each domain's feature space
    # src_l_vecs_list = get_latent_vecs_list(
    #     model,
    #     all_class_files_list=src_l_all_class_files_list,
    #     classes=config['main']['classes'],
    #     input_shape=input_shape,
    #     batch_size=args.batch_size,
    #     output_dir=None
    # )
    # if is_pretrained_model is False:
    #     trg_l_vecs_list = get_latent_vecs_list(
    #         model,
    #         all_class_files_list=trg_l_all_class_files_list,
    #         classes=config['main']['classes'],
    #         input_shape=input_shape,
    #         batch_size=args.batch_size,
    #         output_dir=None
    #     )
    # else:
    #     trg_l_vecs_list = [[], [], []]
    # trg_unl_vecs_list = get_latent_vecs_list(
    #     model,
    #     all_class_files_list=trg_unl_all_class_files_list,
    #     classes=config['main']['classes'],
    #     input_shape=input_shape,
    #     batch_size=args.batch_size,
    #     output_dir=None
    # )

    # other_vecs_list = get_latent_vecs_list(
    #     model,
    #     all_class_files_list=other_all_class_files_list,
    #     classes=config['main']['classes'],
    #     input_shape=input_shape,
    #     batch_size=args.batch_size,
    #     output_dir=None
    # )

    if is_pretrained_model:
        vecs_list_title = f"/home/kengoaraki/Project/DA/AL_SSDA_WSI/plot_miccai/output/ltrg_{l_trg_set}_{l_trg_wsi}_cv{cv_num}_init"
    else:
        vecs_list_title = f"/home/kengoaraki/Project/DA/AL_SSDA_WSI/plot_miccai/output/ltrg_{l_trg_set}_{l_trg_wsi}_cv{cv_num}"
    # np.savez(
    #     vecs_list_title,
    #     src_l_vecs_list,
    #     trg_l_vecs_list,
    #     trg_unl_vecs_list,
    #     other_vecs_list
    # )
    vecs_lists = \
        np.load(f"{vecs_list_title}.npz", allow_pickle=True)
    src_l_vecs_list = vecs_lists['arr_0']
    trg_l_vecs_list = vecs_lists['arr_1']
    trg_unl_vecs_list = vecs_lists['arr_2']
    other_vecs_list = vecs_lists['arr_3']

    logging.info("plot feature space")
    # # pca
    # plot_feature_space_miccai(
    #     src_l_vecs_list=src_l_vecs_list,
    #     trg_l_vecs_list=trg_l_vecs_list,
    #     trg_unl_vecs_list=trg_unl_vecs_list,
    #     other_vecs_list=other_vecs_list,
    #     method="pca",
    #     output_dir=args.output_dir,
    #     title=title
    # )

    # tsne
    plot_feature_space_miccai(
        src_l_vecs_list=src_l_vecs_list,
        trg_l_vecs_list=trg_l_vecs_list,
        trg_unl_vecs_list=trg_unl_vecs_list,
        other_vecs_list=other_vecs_list,
        method="tsne",
        output_dir=args.output_dir,
        title=title
    )

    # # umap
    # plot_feature_space(
    #     src_l_vecs_list=src_l_vecs_list,
    #     trg_l_vecs_list=trg_l_vecs_list,
    #     trg_unl_vecs_list=trg_unl_vecs_list,
    #     method="umap",
    #     output_dir=args.output_dir,
    #     title=title
    # )


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # config_path = "../ST_MICCAI/config_st_cl[0, 1, 2]_valt20_pretrained.yaml"
    config_path = "./ST_MICCAI/config_st_cl[0, 1, 2]_valt20_pretrained.yaml"

    cv_num = 0

    # main(config_path=config_path, l_trg_wsi="03_G144", l_trg_set="top", cv_num=cv_num)
    # main(config_path=config_path, l_trg_wsi="03_G144", l_trg_set="top", cv_num=cv_num, is_pretrained_model=True)

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # main(config_path=config_path, l_trg_wsi="03_G109-1", l_trg_set="btm", cv_num=cv_num)

    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    main(config_path=config_path, l_trg_wsi="03_G177", l_trg_set="med", cv_num=cv_num)
