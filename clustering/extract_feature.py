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


# To get files from one-class
def get_files_WSI_oneclass(
    wsi: str, classes: list, imgs_dir: str, cl_idx: int = 0
):
    cl = classes[cl_idx]
    sub_classes = []
    if isinstance(cl, list):
        for sub_cl in cl:
            sub_classes.append(sub_cl)
    else:
        sub_classes.append(cl)

    re_pattern = re.compile("|".join([f"/{i}/" for i in sub_classes]))

    tmp_file_list = \
        [
            p
            for p in glob.glob(imgs_dir + f"*/{wsi}_*/*.png", recursive=True)
            if bool(re_pattern.search(p))
        ]
    # tmp_list = []
    # for i in range(len(sub_classes)):
    #     sub_cl = sub_classes[i]
    #     tmp_list += [p for p in tmp_file_list if f"/{sub_cl}/" in p]
    # file_list = tmp_list

    file_list = tmp_file_list

    logging.info(f"[{wsi}] cl-{cl}, img_num: {len(file_list)}")
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


def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description="Feature space")
    parser.add_argument(
        "--title",
        type=str,
        default="srcMF0003_cv2",
        metavar="N",
        help="title of this project",
    )
    parser.add_argument(
        "--weight-path",
        type=str,
        default="/mnt/secssd/AL_SSDA_WSI_MICCAI_srcMF0003_strage/s_result/checkpoints/s_MF0003_[0, 1, 2]/cv2_epoch2.pth",
        metavar="N",
        help="weight_path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mnt/secssd/AL_SSDA_WSI_MICCAI_srcMF0003_strage/cluster_entropy/MF0012/npy/",
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
        "--facility",
        type=str,
        default="MF0012",
        metavar="N",
        help="facility name",
    )
    parser.add_argument(
        "--classes", type=list, default=[0, 1, 2], metavar="N", help="classes"
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
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--seed", type=int, default=0, metavar="S", help="random seed (default: 1)"
    )
    args = parser.parse_args()
    return args


# For multiple facilitys
def main():
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
    if args.facility == "MF0003":
        wsi_list = joblib.load(
            f"/mnt/secssd/AL_SSDA_WSI_strage/dataset/{args.facility}/trg_l_wsi.jb"
        )
        wsi_list += joblib.load(
            f"/mnt/secssd/AL_SSDA_WSI_strage/dataset/{args.facility}/trg_unl_wsi.jb"
        )
        wsi_list += joblib.load(
            f"/mnt/secssd/AL_SSDA_WSI_strage/dataset/{args.facility}/valid_wsi.jb"
        )
    elif args.facility == "MF0012":
        wsi_list = joblib.load(
            f"/mnt/secssd/AL_SSDA_WSI_strage/dataset/{args.facility}/cv0_train_{args.facility}_wsi.jb"
        )
        wsi_list += joblib.load(
            f"/mnt/secssd/AL_SSDA_WSI_strage/dataset/{args.facility}/cv0_valid_{args.facility}_wsi.jb"
        )
        wsi_list += joblib.load(
            f"/mnt/secssd/AL_SSDA_WSI_strage/dataset/{args.facility}/cv0_test_{args.facility}_wsi.jb"
        )
    else:
        sys.exit(f"{args.facility} is invalid!")

    # For Baseline and ImageNet
    model = resnet50_midlayer(
        num_classes=len(args.classes), weight_path=args.weight_path
    ).to(device=device)

    for wsi in wsi_list:
        logging.info(f"===== {wsi} =====")
        for cl_idx in range(len(args.classes)):
            file_list = get_files_WSI_oneclass(
                wsi,
                args.classes,
                f"{args.dataset_dir}{args.facility}/",
                cl_idx=cl_idx,
            )

            if len(file_list) > 0:
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
                    f"{args.output_dir}{args.title}_{args.facility}_{wsi}_cl{args.classes[cl_idx]}",
                    latent_vecs,
                )
            else:
                logging.info(f"No files: cl{args.classes[cl_idx]}")


if __name__ == "__main__":
    main()
