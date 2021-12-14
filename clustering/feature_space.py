import os
import sys
import logging
import argparse
import re
import glob
from torch import nn
import numpy as np
import torch
import joblib
import random

from tqdm import tqdm

# from scipy.io import savemat

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from S.dataset import WSI
from S.model import build_model
from visualyze.distribution import plot_feature_space_domains
# from visualyze.utils import imscatter


class resnet50_midlayer(nn.Module):
    def __init__(self, num_classes=4, weight_path=None):
        super(resnet50_midlayer, self).__init__()
        org_model = build_model("resnet50", num_classes, pretrained=True)

        if weight_path is not None:
            org_model.load_state_dict(torch.load(weight_path, map_location="cuda"))
        self.encoder = nn.Sequential(*(list(org_model.children())[:-1]))

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 2048)
        return x


def get_sub_classes(classes):
    # classesからsub-classを取得
    sub_cl_list = []
    for idx in range(len(classes)):
        cl = classes[idx]
        if isinstance(cl, list):
            for sub_cl in cl:
                sub_cl_list.append(sub_cl)
        else:
            sub_cl_list.append(cl)
    return sub_cl_list


# To get N-files from each sub classes
def get_files_equal(wsi_list: list, classes: list, imgs_dir: str, N: int = 250):
    sub_classes = get_sub_classes(classes)
    re_pattern = re.compile("|".join([f"/{i}/" for i in get_sub_classes(classes)]))

    tmp_file_list, file_list = [], []
    for wsi in wsi_list:
        tmp_file_list.extend(
            [
                p
                for p in glob.glob(imgs_dir + f"*/{wsi}_*/*.png", recursive=True)
                if bool(re_pattern.search(p))
            ]
        )
    for i in range(len(sub_classes)):
        sub_cl = sub_classes[i]
        tmp_list = [p for p in tmp_file_list if f"/{sub_cl}/" in p]
        file_list.extend(random.sample(tmp_list, N))
    logging.info(f"img_num: {len(file_list)}")
    return file_list


# To get files from one-class
def get_files_oneclass(
    wsi_list: list, classes: list, imgs_dir: str, N: int = 250, cl_idx: int = 0
):
    cl = classes[cl_idx]
    sub_classes = []
    if isinstance(cl, list):
        for sub_cl in cl:
            sub_classes.append(sub_cl)
    else:
        sub_classes.append(cl)

    re_pattern = re.compile("|".join([f"/{i}/" for i in sub_classes]))

    tmp_file_list = []
    for wsi in wsi_list:
        tmp_file_list.extend(
            [
                p
                for p in glob.glob(imgs_dir + f"*/{wsi}_*/*.png", recursive=True)
                if bool(re_pattern.search(p))
            ]
        )

    tmp_list = []
    for i in range(len(sub_classes)):
        sub_cl = sub_classes[i]
        tmp_list += [p for p in tmp_file_list if f"/{sub_cl}/" in p]

    # file_list = tmp_list

    if len(tmp_list) <= N:
        file_list = tmp_list
        logging.info(f"img_num: {len(file_list)} is less than N={N}")
    else:
        file_list = random.sample(tmp_list, N)

    logging.info(f"cl_{cl}, img_num: {len(file_list)}")
    return file_list


def get_feature(model, device, test_loader, n_data, mode="None"):
    model.eval()
    init_flag = True
    with torch.no_grad():
        with tqdm(total=n_data, unit="img") as pbar:
            for batch in test_loader:
                data, target = batch["image"], batch["label"]
                data, target = data.to(device), target.to(device)
                if mode == "ADA":
                    latent_vecs = model(data, mode="feature")
                else:
                    latent_vecs = model(data)
                pbar.update(target.shape[0])
                latent_vecs = latent_vecs.cpu()
                target = target.cpu()
                if init_flag:
                    latent_vecs_stack = latent_vecs
                    target_stack = target
                    init_flag = False
                else:
                    latent_vecs_stack = torch.cat((latent_vecs_stack, latent_vecs), 0)
                    target_stack = torch.cat((target_stack, target), 0)
    return latent_vecs_stack, target_stack


# # For single domain
# def main():
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(levelname)s: %(message)s'
#     )
#     logging.info("initialize...")

#     # Training settings
#     parser = argparse.ArgumentParser(description='Feature space')
#     parser.add_argument('--weight-path', type=str, default=None, metavar='N',
#                         help='weight_path')
#     parser.add_argument('--dataset-dir', type=str, default="/mnt/ssdsub1/ADA_strage/mnt2/", metavar='N',
#                         help='img dir of dataset')
#     parser.add_argument('--domain_name', type=str, default="MF0012", metavar='N',
#                         help='domain name')
#     parser.add_argument('--data-attr', type=str, default="train", metavar='N',
#                         help='data attribute (default: train)')
#     parser.add_argument('--each-sample-N', type=int, default=200, metavar='N',
#                         help='Each class-sample num')
#     parser.add_argument('--input-shape', type=tuple, default=(256, 256), metavar='N',
#                         help='input-shape of patch img')
#     parser.add_argument('--classes', type=list, default=[2, [1, 3]], metavar='N',
#                         help='classes')
#     parser.add_argument('--batch-size', type=int, default=64, metavar='N',
#                         help='input batch size for training (default: 64)')
#     parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
#                         help='input batch size for testing (default: 64)')
#     parser.add_argument('--epochs', type=int, default=10, metavar='N',
#                         help='number of epochs to train (default: 10)')
#     parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
#                         help='learning rate (default: 0.01)')
#     parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
#                         help='SGD momentum (default: 0.5)')
#     parser.add_argument('--no-cuda', action='store_true', default=False,
#                         help='disables CUDA training')
#     parser.add_argument('--seed', type=int, default=1, metavar='S',
#                         help='random seed (default: 1)')
#     parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                         help='how many batches to wait before logging training\
#                         status')
#     args = parser.parse_args()

#     use_cuda = not args.no_cuda and torch.cuda.is_available()
#     device = torch.device("cuda" if use_cuda else "cpu")
#     kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

#     # fix seed
#     torch.manual_seed(args.seed)
#     random.seed(args.seed)
#     np.random.seed(args.seed)

#     # load dataset
#     logging.info("load file list...")
#     wsi_list = joblib.load(
#         f"/mnt/ssdsub1/ADA_strage/result/dataset/source_{args.domain_name}/cv0_{args.data_attr}_source-{args.domain_name}_wsi.jb"
#     )

#     file_list = get_files_equal(wsi_list, args.classes, f"{args.dataset_dir}{args.domain_name}/", N=args.each_sample_N)
#     random.shuffle(file_list)  # To avoid one class's point overlapping another ones

#     logging.info("set test_loader...")
#     transform = {'Resize': True, 'HFlip': False, 'VFlip': False}
#     dataset = WSI(
#         file_list,
#         args.classes,
#         args.input_shape,
#         transform,
#         is_pred=False
#     )
#     test_loader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=args.test_batch_size,
#         shuffle=False,
#         **kwargs
#     )

#     logging.info("set model...")
#     model = resnet50_midlayer(
#         num_classes=len(args.classes),
#         weight_path=args.weight_path).to(device=device)

#     logging.info("get latent_vecs...")
#     latent_vecs, targets = get_feature(model, device, test_loader, len(dataset))
#     latent_vecs, targets = latent_vecs.numpy(), targets.numpy()
#     logging.info(f"latent_vecs: {latent_vecs.shape}, targets_vecs: {targets.shape}")

#     logging.info("plot feature space")
#     cmap = colormap(len(args.classes), colors)

#     # tsne
#     plot_feature_space(latent_vecs, targets, cmap, method="tsne")
#     # pca
#     plot_feature_space(latent_vecs, targets, cmap, method="pca")
#     # umap
#     plot_feature_space(latent_vecs, targets, cmap, method="umap")
#     # direct
#     plot_feature_space(latent_vecs, targets, cmap, method="direct")

#     logging.info("plot feature space with img")
#     zoom = 0.1
#     # tsne
#     plot_img_feature_space(
#         latent_vecs, targets, file_list, method="tsne", zoom=zoom)
#     # pca
#     plot_img_feature_space(
#         latent_vecs, targets, file_list, method="pca", zoom=zoom)
#     # umap
#     plot_img_feature_space(
#         latent_vecs, targets, file_list, method="umap", zoom=zoom)
#     # # direct
#     # plot_img_feature_space(
#     #     latent_vecs, targets, file_list, method="direct", zoom=zoom)

#     logging.info("plot feature space with img")
#     zoom = 0.2
#     # tsne
#     plot_img_feature_space(
#         latent_vecs, targets, file_list, method="tsne", zoom=zoom)
#     # pca
#     plot_img_feature_space(
#         latent_vecs, targets, file_list, method="pca", zoom=zoom)
#     # umap
#     plot_img_feature_space(
#         latent_vecs, targets, file_list, method="umap", zoom=zoom)
#     # # direct
#     # plot_img_feature_space(
#     #     latent_vecs, targets, file_list, method="direct", zoom=zoom)


def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description="Feature space")
    parser.add_argument(
        "--title",
        type=str,
        default="MF0012_resnet50_Adam_batch32_shape[256, 256]_cl[0, 1, 2]_cv0",
        metavar="N",
        help="title of this project",
    )
    parser.add_argument(
        "--weight-path",
        type=str,
        default="/mnt/secssd/SSDA_Annot_WSI_strage/result/checkpoints/MF0012_[0, 1, 2]/cv0_resnet50_cl[0, 1, 2]_best_val_recall_epoch3.pth",
        metavar="N",
        help="weight_path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mnt/secssd/SSDA_Annot_WSI_strage/result/analysis/",
        metavar="N",
        help="dir of visualizing results",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="/mnt/secssd/SSDA_Annot_WSI_strage/mnt2/",
        metavar="N",
        help="img dir of dataset",
    )
    parser.add_argument(
        "--domain_names",
        type=list,
        default=["MF0012", "MF0003"],
        metavar="N",
        help="domain name list",
    )
    parser.add_argument(
        "--data-attr", type=str, default="train", metavar="N", help="data attribute"
    )
    parser.add_argument(
        "--classes", type=list, default=[0, 1, 2], metavar="N", help="classes"
    )
    parser.add_argument(
        "--domain_labels",
        type=list,
        default=["MF0012_train", "MF0003_all"],
        metavar="N",
        help="domain name list",
    )
    parser.add_argument(
        "--cl_labels",
        type=list,
        default=["Non-Neop.", "HSIL", "LSIL"],
        metavar="N",
        help="domain name list",
    )
    parser.add_argument(
        "--each-sample-N",
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
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    args = parser.parse_args()
    return args


# For multiple domains
def main2():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = get_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 0, "pin_memory": True} if use_cuda else {}

    # fix seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # load dataset
    logging.info("load wsi list...")
    domain1_wsi_list = joblib.load(
        f"/mnt/secssd/SSDA_Annot_WSI_strage/dataset/{args.domain_names[0]}/cv0_{args.data_attr}_{args.domain_names[0]}_wsi.jb"
    )

    domain2_wsi_list = joblib.load(
        f"/mnt/secssd/SSDA_Annot_WSI_strage/dataset/{args.domain_names[1]}/cv0_train_{args.domain_names[1]}_wsi.jb"
    )
    domain2_wsi_list += joblib.load(
        f"/mnt/secssd/SSDA_Annot_WSI_strage/dataset/{args.domain_names[1]}/cv0_valid_{args.domain_names[1]}_wsi.jb"
    )
    domain2_wsi_list += joblib.load(
        f"/mnt/secssd/SSDA_Annot_WSI_strage/dataset/{args.domain_names[1]}/cv0_test_{args.domain_names[1]}_wsi.jb"
    )

    all_domain_wsi_list = [domain1_wsi_list, domain2_wsi_list]

    logging.info("set model...")

    # For Baseline and ImageNet
    model = resnet50_midlayer(
        num_classes=len(args.classes), weight_path=args.weight_path
    ).to(device=device)

    # visualize each domain's feature space
    latent_vecs_list = []
    for idx, wsi_list in enumerate(all_domain_wsi_list):
        dataset_latent_vecs_list = []

        if len(wsi_list) > 0:
            for cl_idx in range(len(args.classes)):
                file_list = get_files_oneclass(
                    wsi_list,
                    args.classes,
                    f"{args.dataset_dir}{args.domain_names[idx]}/",
                    N=args.each_sample_N,
                    cl_idx=cl_idx,
                )

                transform = {"Resize": True, "HFlip": False, "VFlip": False}
                dataset = WSI(
                    file_list, args.classes, args.input_shape, transform, is_pred=False
                )
                dataset_loader = torch.utils.data.DataLoader(
                    dataset, batch_size=args.batch_size, shuffle=False, **kwargs
                )

                # logging.info("get latent_vecs...")
                latent_vecs, _ = get_feature(
                    model, device, dataset_loader, len(dataset)
                )
                latent_vecs = latent_vecs.numpy()

                # 保存
                np.save(
                    f"{args.output_dir}resnet50_srcMF0012_{args.data_attr}_cv0_all_{args.domain_names[idx]}_cl{cl_idx}",
                    latent_vecs,
                )
                # mat_latent_vecs = {
                #     "feature_vecs": latent_vecs,
                #     "label": f"{args.domain_names[idx]}_cl{cl_idx}",
                # }
                # savemat(f"{args.output_dir}resnet50_srcMF0012_{args.data_attr}_cv0_all_{args.domain_names[idx]}_cl{cl_idx}.mat", mat_latent_vecs)

                dataset_latent_vecs_list.append(latent_vecs)

            latent_vecs_list.append(dataset_latent_vecs_list)
            logging.info(f"latent_vecs: {latent_vecs.shape}")
        else:
            latent_vecs_list.append([])

    logging.info("plot feature space")
    # tsne
    plot_feature_space_domains(
        latent_vecs_list,
        method="tsne",
        output_dir=args.output_dir,
        title=args.title + "_" + args.data_attr,
        domain_labels=args.domain_labels,
        cl_labels=args.cl_labels,
    )
    # pca
    plot_feature_space_domains(
        latent_vecs_list,
        method="pca",
        output_dir=args.output_dir,
        title=args.title + "_" + args.data_attr,
        domain_labels=args.domain_labels,
        cl_labels=args.cl_labels,
    )
    # umap
    plot_feature_space_domains(
        latent_vecs_list,
        method="umap",
        output_dir=args.output_dir,
        title=args.title + "_" + args.data_attr,
        domain_labels=args.domain_labels,
        cl_labels=args.cl_labels,
    )


if __name__ == "__main__":
    main2()

    # from scipy.io import loadmat
    # test = np.load("/mnt/secssd/SSDA_Annot_WSI_strage/analysis/FeatureData/resnet50_srcMF0012_train_cv0_MF0003_cl0.npy")
    # test_mat = loadmat("/mnt/secssd/SSDA_Annot_WSI_strage/analysis/FeatureData/resnet50_srcMF0012_train_cv0_MF0003_cl0.mat")
    # print(test)
