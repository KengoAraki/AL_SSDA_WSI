import os
import sys
import torch
import random
import numpy as np
import logging
import argparse
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import PCA

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from visualyze2.feature import extract_feature, WSI
from visualyze2.plot import merge_vecs_list
from visualyze2.fe_model import MME_resnet50_midlayer, Prototype
from visualyze2.vis_utils import get_files_oneclass, markers, pltcolor, set_colors
from utils.return_dataset_WSI import get_wsi_list


def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description="Feature space")
    parser.add_argument(
        "--title",
        type=str,
        default="MME_MF0012_to_MF0003_cl[0, 1, 2]_cv0",
        metavar="N",
        help="title of this project",
    )
    parser.add_argument(
        '--method',
        type=str,
        default='pretrainedMME',
        choices=['S+T', 'ENT', 'MME', 'pretrainedMME', 'weightedMME'],
        help='MME is proposed method, ENT is entropy minimization, S+T is training only on labeled examples')
    parser.add_argument(
        "--G_weight_dir",
        type=str,
        default="/home/kengoaraki/Project/DA/SSDA_MME_Saito_WSI/output/save_model_ssda/pretrainedMME/03_G144/cv0/",
        metavar="N",
        help="G_weight_dir",
    )
    parser.add_argument(
        "--F1_weight_dir",
        type=str,
        default="/home/kengoaraki/Project/DA/SSDA_MME_Saito_WSI/output/save_model_ssda/pretrainedMME/03_G144/cv0/",
        metavar="N",
        help="F1_weight_dir",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/kengoaraki/Project/DA/SSDA_MME_Saito_WSI/output/result/pretrainedMME/03_G144/",
        metavar="N",
        help="dir of visualizing results",
    )
    parser.add_argument(
        "--classes", type=list, default=[0, 1, 2], metavar="N", help="classes"
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
        "--input-shape",
        type=tuple,
        default=(256, 256),
        metavar="N",
        help="input-shape of patch img",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )

    parser.add_argument('--src_imgs_dir', type=str, default="/mnt/secssd/SSDA_Annot_WSI_strage/mnt2/MF0012/", metavar='N',
                        help='source images directory')
    parser.add_argument('--trg_imgs_dir', type=str, default="/mnt/secssd/SSDA_Annot_WSI_strage/mnt2/MF0003/", metavar='N',
                        help='target images directory')
    parser.add_argument('--src_train_jb_path', type=str, default="/mnt/secssd/SSDA_Annot_WSI_strage/dataset/MF0012/cv0_train_MF0012_wsi.jb", metavar='N',
                        help='source train joblib path')
    parser.add_argument('--trg_train_jb_path', type=str, default="/mnt/secssd/SSDA_Annot_WSI_strage/dataset/MF0003/cv0_train_MF0003_wsi.jb", metavar='N',
                        help='target train joblib path')
    parser.add_argument('--trg_val_jb_path', type=str, default="/mnt/secssd/SSDA_Annot_WSI_strage/dataset/MF0003/cv0_valid_MF0003_wsi.jb", metavar='N',
                        help='target valid joblib path')
    parser.add_argument('--trg_test_jb_path', type=str, default="/mnt/secssd/SSDA_Annot_WSI_strage/dataset/MF0003/cv0_test_MF0003_wsi.jb", metavar='N',
                        help='target test joblib path')
    parser.add_argument('--trg_val_wsis', type=list, default=["03_G170", "03_G142", "03_G143"], metavar='N',
                        help='target valid wsi list')

    parser.add_argument('--trg_selected_wsi', type=str, default="03_G144", metavar='N',
                        help='trg_selected_wsi (highest entropy)')
    # parser.add_argument('--trg_selected_wsi', type=str, default="03_G293", metavar='N',
    #                     help='trg_selected_wsi (middle entropy)')
    # parser.add_argument('--trg_selected_wsi', type=str, default="03_G109-1", metavar='N', help='trg_selected_wsi (lowest entropy)')
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    args = parser.parse_args()
    return args


def get_latent_vecs(
    model,
    file_list: list,
    classes: list = [0, 1, 2],
    input_shape: tuple = (256, 256),
    batch_size: int = 32,
):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 0, "pin_memory": True} if use_cuda else {}

    if len(file_list) > 0:
        transform = {"Resize": True, "HFlip": False, "VFlip": False}
        dataset = WSI(
            file_list, classes, input_shape, transform, is_pred=False
        )
        dataset_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, **kwargs
        )

        latent_vecs, _ = extract_feature(
            model, device, dataset_loader, len(dataset)
        )
        latent_vecs = latent_vecs.numpy()
    else:  # このクラスのデータが存在しない場合
        latent_vecs = np.array([])
    return latent_vecs


def get_latent_vecs_list(
    model,
    file_lists: list,
    classes: list = [0, 1, 2],
    input_shape: tuple = (256, 256),
    batch_size: int = 32,
    output_dir: str = None,
    title: str = "latent_vecs"
):
    latent_vecs_list = []
    for cl_idx, file_list in zip(classes, file_lists):
        # クラスごとに特徴量ベクトルを取得
        latent_vecs = get_latent_vecs(
            model,
            file_list=file_list,
            classes=classes,
            input_shape=input_shape, batch_size=batch_size,
        )
        # 特徴ベクトルを保存
        if output_dir is not None:
            np.save(
                f"{output_dir}{title}_cl{cl_idx}",
                latent_vecs,
            )

        latent_vecs_list.append(latent_vecs)
    logging.info(f"latent_vecs: {latent_vecs.shape}")
    return latent_vecs_list


def get_file_lists(
    wsi_list: list,
    imgs_dir: str,
    sample_N: int = 500,
    classes: list = [0, 1, 2],
):
    file_lists = []
    for cl_idx in range(len(classes)):
        file_list = get_files_oneclass(
            wsi_list=wsi_list,
            classes=classes,
            imgs_dir=imgs_dir,
            N=sample_N,
            cl_idx=cl_idx,
        )
        file_lists.append(file_list)
    return file_lists


def get_pca_feature(x, pca_model=None):
    if pca_model is None:
        print("initialize pca_model")
        pca_model = PCA(n_components=2, random_state=0)
        pca_model = pca_model.fit(x)

    x_embedded = pca_model.transform(x)
    return x_embedded, pca_model


# labeled_src, labeled_trg, unlebeled_trg, prototypeをplot
def plot_pca_MME(
    src_l_vecs_list: list,
    trg_l_vecs_list: list,
    trg_unl_vecs_list: list,
    prototype_vecs_list: list,
    method: str = "tsne",
    output_dir: str = None,
    title: str = "feature_space",
    pca_model=None,
):
    start_time = time.time()

    print(f"Visualize features with {method}")
    plt.figure(figsize=(9, 6))
    set_dict = {"prototype": 0, "src_l": 1, "trg_l": 2, "trg_unl": 3}

    # 各特徴量をマージ
    x_all, cl_idx_list, num_x_list, set_idx_list = \
        merge_vecs_list(src_l_vecs_list, set_idx=set_dict['src_l'])
    x_all, cl_idx_list, num_x_list, set_idx_list = \
        merge_vecs_list(trg_unl_vecs_list, x_all, cl_idx_list, num_x_list, set_idx_list, set_idx=set_dict['trg_unl'])
    x_all, cl_idx_list, num_x_list, set_idx_list = \
        merge_vecs_list(trg_l_vecs_list, x_all, cl_idx_list, num_x_list, set_idx_list, set_idx=set_dict['trg_l'])
    x_all, cl_idx_list, num_x_list, set_idx_list = \
        merge_vecs_list(prototype_vecs_list, x_all, cl_idx_list, num_x_list, set_idx_list, set_idx=set_dict['prototype'])

    # 次元圧縮
    x_embedded, model = get_pca_feature(x_all, pca_model)

    total_num = 0
    for i, (cl_idx, num_x, set_idx) in enumerate(
        zip(cl_idx_list, num_x_list, set_idx_list)
    ):
        # データが存在しない場合はスキップ
        if num_x < 1:
            continue

        # color (set別)
        c = [pltcolor(label=set_idx, cols=set_colors)]
        # marker (class別)
        marker = markers[cl_idx]

        # plt.scatter(
        #     x_embedded[total_num: total_num + num_x, 0],
        #     x_embedded[total_num: total_num + num_x, 1],
        #     c=c,
        #     marker=marker,
        #     label=None,
        #     alpha=0.8,
        #     linewidth=0,
        #     edgecolors='face'
        # )

        if set_idx == set_dict['prototype']:
            plt.scatter(
                x_embedded[total_num: total_num + num_x, 0] / 3,
                x_embedded[total_num: total_num + num_x, 1] / 3,
                c=c,
                marker=marker,
                label=None,
                alpha=0.8,
                linewidth=0,
                edgecolors='face'
            )
        else:
            plt.scatter(
                x_embedded[total_num: total_num + num_x, 0],
                x_embedded[total_num: total_num + num_x, 1],
                c=c,
                marker=marker,
                label=None,
                alpha=0.8,
                linewidth=0,
                edgecolors='face'
            )
        total_num += num_x

    # 凡例用 (class)
    plt.scatter([], [], c="silver", alpha=1, marker=markers[0], label="Non-Neop")
    plt.scatter([], [], c="silver", alpha=1, marker=markers[2], label="LSIL")
    plt.scatter([], [], c="silver", alpha=1, marker=markers[1], label="HSIL")
    # 凡例用 (set)
    c = [pltcolor(label=set_dict['prototype'], cols=set_colors)]
    plt.scatter([], [], c=c, alpha=1, marker=markers[0], label="Prototype")
    c = [pltcolor(label=set_dict['src_l'], cols=set_colors)]
    plt.scatter([], [], c=c, alpha=1, marker=markers[0], label="Labeled Source")
    c = [pltcolor(label=set_dict['trg_l'], cols=set_colors)]
    plt.scatter([], [], c=c, alpha=1, marker=markers[0], label="Labeled Target")
    c = [pltcolor(label=set_dict['trg_unl'], cols=set_colors)]
    plt.scatter([], [], c=c, alpha=1, marker=markers[0], label="Unlabeled Target")

    elapsed_time = time.time() - start_time
    logging.info(f"elapsed_time: {elapsed_time:.1f} [sec]")

    plt.title(f"{title} with {method}")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    if output_dir is not None:
        plt.savefig(
            f"{output_dir}{title}_{method}.png",
            format="png",
            dpi=300,
            bbox_inches="tight",
        )
    # plt.show()
    plt.clf()
    plt.close()
    return model


# For multiple domains
def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = get_args()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # fix seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # load wsi_list
    logging.info("load wsi list...")
    src_l_wsis, trg_l_wsis, _, trg_unl_wsis = \
        get_wsi_list(
            trg_selected_wsi=args.trg_selected_wsi,
            trg_val_wsis=args.trg_val_wsis,
            src_train_jb_path=args.src_train_jb_path,
            trg_train_jb_path=args.trg_train_jb_path,
            trg_val_jb_path=args.trg_val_jb_path,
            trg_test_jb_path=args.trg_test_jb_path
        )

    src_l_file_lists = get_file_lists(
        wsi_list=src_l_wsis,
        imgs_dir=args.src_imgs_dir,
        sample_N=args.sample_N,
        classes=args.classes,
    )
    trg_l_file_lists = get_file_lists(
        wsi_list=trg_l_wsis,
        imgs_dir=args.trg_imgs_dir,
        sample_N=args.sample_N,
        classes=args.classes,
    )
    trg_unl_file_lists = get_file_lists(
        wsi_list=trg_unl_wsis,
        imgs_dir=args.trg_imgs_dir,
        sample_N=args.sample_N,
        classes=args.classes,
    )

    pca_model = None
    for iter_idx in range(0, 24001, 100):
        if iter_idx == 0:
            iter_idx = 1

        title = f"{args.title}_iter{iter_idx}"
        logging.info(f"=== {title} ===")
        G_weight_path = os.path.join(
            args.G_weight_dir,
            f"G_{args.method}_MF0012_to_MF0003_iter{iter_idx}.pth"
        )
        F1_weight_path = os.path.join(
            args.F1_weight_dir,
            f"F1_{args.method}_MF0012_to_MF0003_iter{iter_idx}.pth"
        )

        # load model
        logging.info("set model...")
        model = MME_resnet50_midlayer(
            num_classes=len(args.classes),
            G_weight_path=G_weight_path,
            F1_weight_path=F1_weight_path,
        ).to(device=device)

        # visualize each domain's feature space
        src_l_vecs_list = get_latent_vecs_list(
            model,
            file_lists=src_l_file_lists,
            classes=args.classes,
            input_shape=args.input_shape,
            batch_size=args.batch_size,
            output_dir=None
        )
        trg_l_vecs_list = get_latent_vecs_list(
            model,
            file_lists=trg_l_file_lists,
            classes=args.classes,
            input_shape=args.input_shape,
            batch_size=args.batch_size,
            output_dir=None
        )
        trg_unl_vecs_list = get_latent_vecs_list(
            model,
            file_lists=trg_unl_file_lists,
            classes=args.classes,
            input_shape=args.input_shape,
            batch_size=args.batch_size,
            output_dir=None
        )
        prototype_vecs_list = Prototype(
            num_classes=len(args.classes),
            F1_weight_path=F1_weight_path
        ).get()

        logging.info("plot feature space")
        # pca
        pca_model = plot_pca_MME(
            src_l_vecs_list=src_l_vecs_list,
            trg_l_vecs_list=trg_l_vecs_list,
            trg_unl_vecs_list=trg_unl_vecs_list,
            prototype_vecs_list=prototype_vecs_list,
            method="pca",
            output_dir=args.output_dir,
            title=title + "_" + args.trg_selected_wsi,
            pca_model=pca_model
        )

        # # tsne
        # plot_feature_space_MME(
        #     src_l_vecs_list=src_l_vecs_list,
        #     trg_l_vecs_list=trg_l_vecs_list,
        #     trg_unl_vecs_list=trg_unl_vecs_list,
        #     prototype_vecs_list=prototype_vecs_list,
        #     method="tsne",
        #     output_dir=args.output_dir,
        #     title=title + "_" + args.trg_selected_wsi
        # )


if __name__ == "__main__":
    main()
