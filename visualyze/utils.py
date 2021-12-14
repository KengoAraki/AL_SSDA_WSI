import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


# Domain識別用
markers = ['o', '*', '^', 'x', '+', 'P', 's']

# 各classのcolor
colors = [
    (200, 200, 200),
    (255, 0, 0),
    (255, 255, 0),
    (0, 255, 0),
    (0, 255, 255),
    (0, 0, 255),
]

# 各domainの各classのcolor
colors_domains = [
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


def colormap(N: int = 5, cols: list = colors):
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
    cmap = ListedColormap(tmp_cols, name="custom", N=N)
    return cmap


def edgecolor(label: int, cols: list):
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
            bboxprops=dict(edgecolor=edgecolor(lb)))
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists
