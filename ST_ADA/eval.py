import os
import sys
import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
# import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from S.eval import get_confusion_matrix, plot_confusion_matrix, convert_plt2nd, eval_metrics


def make_confusion_matrix(preds, labels, cm=None):
    preds = nn.Softmax(dim=1)(preds)
    if cm is None:
        cm = get_confusion_matrix(preds, labels)
    else:
        cm += get_confusion_matrix(preds, labels)
    return cm


def logging_cm(cm, classes, epoch, normalize, mode, obj, writer):
    cm_plt = plot_confusion_matrix(cm, classes, normalize=normalize)
    cm_nd = convert_plt2nd(cm_plt)
    if normalize:
        title = f'confusion_matrix/{mode}_{obj}'
    else:
        title = f'confusion_matrix_nn/{mode}_{obj}'
    writer.add_image(
        title,
        cm_nd,
        global_step=epoch,
        dataformats='HWC'
    )
    plt.clf()
    plt.close()
    return writer


def tensorboard_logging(
    writer,
    epoch,
    loss_EC=None,
    loss_A=None,
    loss_D=None,
    l_cm=None,
    unl_cm=None,
    d_cm=None,
    classes=None,
    mode="train"
):
    if loss_EC is not None:
        writer.add_scalar(f'Loss_EC/{mode}', loss_EC, epoch)

    if loss_A is not None:
        writer.add_scalar(f'Loss_A/{mode}', loss_A, epoch)

    if loss_D is not None:
        writer.add_scalar(f'Loss_D/{mode}', loss_D, epoch)

    if l_cm is not None:
        writer = logging_cm(l_cm, classes, epoch, normalize=True, mode=mode, obj="clf-labeled", writer=writer)
        writer = logging_cm(l_cm, classes, epoch, normalize=False, mode=mode, obj="clf-labeled", writer=writer)
        l_metrics = eval_metrics(l_cm)
        writer.add_scalar(f'Accuracy (l)/{mode}', l_metrics['accuracy'], epoch)
        writer.add_scalar(f'Precision (l)/{mode}', l_metrics['precision'], epoch)
        writer.add_scalar(f'Recall (l)/{mode}', l_metrics['recall'], epoch)
        writer.add_scalar(f'mIoU (l)/{mode}', l_metrics['mIoU'], epoch)
        writer.add_scalar(f'F1 (l)/{mode}', l_metrics['f1'], epoch)

    if unl_cm is not None:
        writer = logging_cm(unl_cm, classes, epoch, normalize=True, mode=mode, obj="clf-unlabeled", writer=writer)
        writer = logging_cm(unl_cm, classes, epoch, normalize=False, mode=mode, obj="clf-unlabeld", writer=writer)
        unl_metrics = eval_metrics(unl_cm)
        writer.add_scalar(f'Accuracy (unl)/{mode}', unl_metrics['accuracy'], epoch)
        writer.add_scalar(f'Precision (unl)/{mode}', unl_metrics['precision'], epoch)
        writer.add_scalar(f'Recall (unl)/{mode}', unl_metrics['recall'], epoch)
        writer.add_scalar(f'mIoU (unl)/{mode}', unl_metrics['mIoU'], epoch)
        writer.add_scalar(f'F1 (unl)/{mode}', unl_metrics['f1'], epoch)

    if d_cm is not None:
        classes_names = ["source", "target"]  # if source_label: 0, target_label: 1
        writer = logging_cm(d_cm, classes_names, epoch, normalize=True, mode=mode, obj="discriminator", writer=writer)
        writer = logging_cm(d_cm, classes_names, epoch, normalize=False, mode=mode, obj="discriminator", writer=writer)
        d_val_metrics = eval_metrics(d_cm)
        writer.add_scalar(f'Accuracy (Discriminator)/{mode}', d_val_metrics['accuracy'], epoch)
        writer.add_scalar(f'Precision (Discriminator)/{mode}', d_val_metrics['precision'], epoch)
        writer.add_scalar(f'Recall (Discriminator)/{mode}', d_val_metrics['recall'], epoch)
        writer.add_scalar(f'mIoU (Discriminator)/{mode}', d_val_metrics['mIoU'], epoch)
        writer.add_scalar(f'F1 (Discriminator)/{mode}', d_val_metrics['f1'], epoch)

    return writer


# ========== for train.py ========== #
# For train sets
def eval_net_train(
    clf_l_preds,
    clf_l_labels,
    clf_unl_preds,
    clf_unl_labels,
    d_src_preds,
    d_src_labels,
    d_trg_preds,
    d_trg_labels,
    clf_l_cm=None,
    clf_unl_cm=None,
    d_cm=None,
):
    # confusion matrix (source)
    clf_l_cm = make_confusion_matrix(clf_l_preds, clf_l_labels, cm=clf_l_cm)
    d_src_cm = make_confusion_matrix(d_src_preds, d_src_labels, cm=None)

    # confusion matrix (target)
    clf_unl_cm = make_confusion_matrix(clf_unl_preds, clf_unl_labels, cm=clf_unl_cm)
    d_trg_cm = make_confusion_matrix(d_trg_preds, d_trg_labels, cm=None)

    d_cm += d_src_cm + d_trg_cm
    return clf_l_cm, clf_unl_cm, d_cm


# for validation sets (target)
def eval_net_trg_val(netE, netD, loader, criterion, device, trg_label=1):
    netE.eval()
    netD.eval()

    n_val = len(loader)  # the number of batch

    total_clf_loss = 0
    total_d_loss = 0

    with tqdm(total=n_val, desc="Validation round", unit="batch", leave=False) as pbar:
        for i, batch in enumerate(loader):
            imgs, clf_labels = batch['image'], batch['label']
            imgs = imgs.to(device=device, dtype=torch.float32)
            clf_labels = clf_labels.to(device=device, dtype=torch.long)

            d_trg_labels = torch.full((imgs.data.size()[0],), trg_label,
                                      dtype=torch.long, device=device)

            with torch.no_grad():
                clf_preds = netE(imgs, mode='class')
                clf_loss = criterion(clf_preds, clf_labels)

                featuremap = netE(imgs, mode='feature')
                featuremap = featuremap.detach()
                d_preds = netD(featuremap)
                d_loss = criterion(d_preds, d_trg_labels)

            total_clf_loss += clf_loss.item()
            total_d_loss += d_loss.item()

            # confusion matrix
            if i == 0:
                clf_cm = make_confusion_matrix(clf_preds, clf_labels, cm=None)
                d_cm = make_confusion_matrix(d_preds, d_trg_labels, cm=None)
            else:
                clf_cm = make_confusion_matrix(clf_preds, clf_labels, cm=clf_cm)
                d_cm = make_confusion_matrix(d_preds, d_trg_labels, cm=d_cm)

            pbar.update()

    netE.train()
    netD.train()
    return total_clf_loss / n_val, total_d_loss / n_val, clf_cm, d_cm


# ========== for train_d.py ========== #
# for train set (train_d)
def eval_net_train_d(
    d_src_preds,
    d_src_labels,
    d_trg_preds,
    d_trg_labels,
    d_cm=None,
):
    # confusion matrix (source)
    d_src_cm = make_confusion_matrix(d_src_preds, d_src_labels, cm=None)

    # confusion matrix (target)
    d_trg_cm = make_confusion_matrix(d_trg_preds, d_trg_labels, cm=None)
    d_cm += d_src_cm + d_trg_cm
    return d_cm


# for validation sets (target, train_d)
def eval_net_trg_val_d(netE, netD, loader, criterion, device, trg_label=1):
    netE.eval()
    netD.eval()

    n_val = len(loader)  # the number of batch

    total_d_loss = 0

    with tqdm(total=n_val, desc="Validation round", unit="batch", leave=False) as pbar:
        for i, batch in enumerate(loader):
            imgs = batch['image'].to(device=device, dtype=torch.float32)
            d_trg_labels = torch.full((imgs.data.size()[0],), trg_label,
                                      dtype=torch.long, device=device)

            with torch.no_grad():
                featuremap = netE(imgs, mode='feature')
                featuremap = featuremap.detach()
                d_preds = netD(featuremap)
                d_loss = criterion(d_preds, d_trg_labels)

            total_d_loss += d_loss.item()

            # confusion matrix
            if i == 0:
                d_cm = make_confusion_matrix(d_preds, d_trg_labels, cm=None)
            else:
                d_cm = make_confusion_matrix(d_preds, d_trg_labels, cm=d_cm)

            pbar.update()

    netE.train()
    netD.train()
    return total_d_loss / n_val, d_cm
