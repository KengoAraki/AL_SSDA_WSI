import os
import sys
import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
# import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from ST_ADA.eval import make_confusion_matrix

# ========== for train.py ========== #
# For train sets
def eval_net_train(
    clf_preds,
    clf_labels,
    d_src_preds,
    d_src_labels,
    d_trg_preds,
    d_trg_labels,
    clf_cm=None,
    d_cm=None,
):
    # confusion matrix (source)
    clf_cm = make_confusion_matrix(clf_preds, clf_labels, cm=clf_cm)
    d_src_cm = make_confusion_matrix(d_src_preds, d_src_labels, cm=None)

    # confusion matrix (target)
    d_trg_cm = make_confusion_matrix(d_trg_preds, d_trg_labels, cm=None)

    d_cm += d_src_cm + d_trg_cm
    return clf_cm, d_cm


# # for validation sets (target)
# def eval_net_trg_val(netE, netD, loader, criterion, device, trg_label=1):
#     netE.eval()
#     netD.eval()

#     n_val = len(loader)  # the number of batch

#     total_clf_loss = 0
#     total_d_loss = 0

#     with tqdm(total=n_val, desc="Validation round", unit="batch", leave=False) as pbar:
#         for i, batch in enumerate(loader):
#             imgs, clf_labels = batch['image'], batch['label']
#             imgs = imgs.to(device=device, dtype=torch.float32)
#             clf_labels = clf_labels.to(device=device, dtype=torch.long)

#             d_trg_labels = torch.full((imgs.data.size()[0],), trg_label,
#                                       dtype=torch.long, device=device)

#             with torch.no_grad():
#                 clf_preds = netE(imgs, mode='class')
#                 clf_loss = criterion(clf_preds, clf_labels)

#                 featuremap = netE(imgs, mode='feature')
#                 featuremap = featuremap.detach()
#                 d_preds = netD(featuremap)
#                 d_loss = criterion(d_preds, d_trg_labels)

#             total_clf_loss += clf_loss.item()
#             total_d_loss += d_loss.item()

#             # confusion matrix
#             if i == 0:
#                 clf_cm = make_confusion_matrix(clf_preds, clf_labels, cm=None)
#                 d_cm = make_confusion_matrix(d_preds, d_trg_labels, cm=None)
#             else:
#                 clf_cm = make_confusion_matrix(clf_preds, clf_labels, cm=clf_cm)
#                 d_cm = make_confusion_matrix(d_preds, d_trg_labels, cm=d_cm)

#             pbar.update()

#     netE.train()
#     netD.train()
#     return total_clf_loss / n_val, total_d_loss / n_val, clf_cm, d_cm


# # for test (target)
# def eval_net_test(netE, loader, criterion, device):
#     netE.eval()

#     n_val = len(loader)  # the number of batch

#     total_clf_loss = 0

#     with tqdm(total=n_val, desc="Validation round", unit="batch", leave=False) as pbar:
#         for i, batch in enumerate(loader):
#             imgs, clf_labels = batch['image'], batch['label']
#             imgs = imgs.to(device=device, dtype=torch.float32)
#             clf_labels = clf_labels.to(device=device, dtype=torch.long)

#             with torch.no_grad():
#                 clf_preds = netE(imgs, mode='class')
#                 clf_loss = criterion(clf_preds, clf_labels)

#             total_clf_loss += clf_loss.item()

#             # confusion matrix
#             if i == 0:
#                 clf_cm = make_confusion_matrix(clf_preds, clf_labels, cm=None)
#             else:
#                 clf_cm = make_confusion_matrix(clf_preds, clf_labels, cm=clf_cm)

#             pbar.update()

#     netE.train()
#     return total_clf_loss / n_val, clf_cm
