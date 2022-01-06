import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import random
import logging
import re
import glob


# Domain識別用
markers = ['o', '*', '^', 'x', '+', 'P', 's']

# 各classのcolor
cl_colors = [
    (200, 200, 200),
    (255, 0, 0),
    (255, 255, 0),
    (0, 255, 0),
    (0, 255, 255),
    (0, 0, 255),
]

# 各domainの各classのcolor
domain_colors = [
    [
        (0, 0, 255),
        (255, 0, 0),
        (0, 255, 0),
    ],
    [
        (0, 100, 255),
        (255, 100, 0),
        (100, 255, 0),
    ],
    [
        (100, 100, 255),
        (255, 100, 100),
        (0, 255, 100),
    ],
]


set_colors = [
    (255, 128, 0),
    (0, 102, 204),
    (102, 178, 255),
]


def pltcolor(label: int, cols: list):
    def calc(cols):
        cs = []
        for c in cols:
            cs.append(
                (round(c[0] / 255, 1),
                 round(c[1] / 255, 1),
                 round(c[2] / 255, 1))
            )
        return cs

    tmp_cols = calc(cols)
    color = tmp_cols[label]
    return color


def imscatter(x, y, labels, image_list, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    im_list = [OffsetImage(plt.imread(str(p)), zoom=zoom) for p in image_list]
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0, lb, im in zip(x, y, labels, im_list):
        # ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        ab = AnnotationBbox(
            im,
            (x0, y0),
            xycoords='data',
            frameon=True,
            pad=0.4,
            bboxprops=dict(edgecolor=pltcolor(lb)))
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists


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

    if len(tmp_list) <= N:
        file_list = tmp_list
        logging.info(f"img_num: {len(file_list)} is less than N={N}")
    else:
        file_list = random.sample(tmp_list, N)

    logging.info(f"cl_{cl}, img_num: {len(file_list)}")
    return file_list


# # To get N-files from each sub classes
# def get_files_equal(wsi_list: list, classes: list, imgs_dir: str, N: int = 250):
#     sub_classes = get_sub_classes(classes)
#     re_pattern = re.compile("|".join([f"/{i}/" for i in get_sub_classes(classes)]))

#     tmp_file_list, file_list = [], []
#     for wsi in wsi_list:
#         tmp_file_list.extend(
#             [
#                 p
#                 for p in glob.glob(imgs_dir + f"*/{wsi}_*/*.png", recursive=True)
#                 if bool(re_pattern.search(p))
#             ]
#         )
#     for i in range(len(sub_classes)):
#         sub_cl = sub_classes[i]
#         tmp_list = [p for p in tmp_file_list if f"/{sub_cl}/" in p]
#         file_list.extend(random.sample(tmp_list, N))
#     logging.info(f"img_num: {len(file_list)}")
#     return file_list
