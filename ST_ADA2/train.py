import os
import sys
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import yaml
import logging
import joblib
import json

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from S.eval import eval_metrics
from S.train import update_best_model, early_stop
from S.util import fix_seed, ImbalancedDatasetSampler, select_optim
from ST_ADA.model import Encoder, Discriminator
from ST_ADA2.dataset import WSIDataset_ST1_ADA2_ValT
from ST_ADA.eval import eval_net_trg_val, tensorboard_logging
from ST_ADA2.eval import eval_net_train


def train_net(
    netE,
    netD,
    clf_l_src_train_data,
    clf_l_trg_train_data,
    d_l_src_train_data,
    d_unl_trg_train_data,
    valid_data,
    unl_trg_test_data,
    device,
    epochs: int = 5,
    batch_size: int = 16,
    optim_name: str = "Adam",
    classes: list = [0, 1, 2],
    checkpoint_dir: str = "checkpoints/",
    writer=None,
    patience: int = 5,
    stop_cond: str = "mIoU",
    cv_num: int = 0,
):
    """
    netE: Encoder network (nn.Module),
    netD: Discriminator network (nn.Module),
    clf_l_src_train_data: labeled source train datasaet (torch.utils.data.Dataset),
    clf_l_trg_train_data: labeled target train datasaet (torch.utils.data.Dataset),
    valid_data: target validation dataset (torch.utils.data.Dataset),
    """
    # class-balanced
    clf_l_src_train_loader = DataLoader(
        clf_l_src_train_data,
        sampler=ImbalancedDatasetSampler(clf_l_src_train_data),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # class-balanced
    clf_l_trg_train_loader = DataLoader(
        clf_l_trg_train_data,
        sampler=ImbalancedDatasetSampler(clf_l_trg_train_data),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # cluster-balanced
    d_l_src_train_loader = DataLoader(
        d_l_src_train_data,
        sampler=ImbalancedDatasetSampler(d_l_src_train_data),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    # cluster-balanced
    d_unl_trg_train_loader = DataLoader(
        d_unl_trg_train_data,
        sampler=ImbalancedDatasetSampler(d_unl_trg_train_data),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        valid_data,
        sampler=None,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    unl_trg_test_loader = DataLoader(
        unl_trg_test_data,
        sampler=None,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    optE = select_optim(optim_name, netE.parameters())
    optD = select_optim(optim_name, netD.parameters())

    criterion = nn.CrossEntropyLoss()

    if stop_cond == "val_loss":
        mode = "min"
    else:
        mode = "max"

    if mode == "min":
        best_model_info = {"epoch": 0, "val": float("inf")}
    elif mode == "max":
        best_model_info = {"epoch": 0, "val": float("-inf")}

    D_src_label = 0
    D_trg_label = 1

    n_train = min(len(clf_l_src_train_loader), len(clf_l_trg_train_loader), len(d_l_src_train_loader), len(d_unl_trg_train_loader))

    for epoch in range(epochs):
        netE.train()
        netD.train()

        # pbar = tqdm(total=num_iter)
        EC_loss_value = 0
        A_loss_value = 0
        D_loss_value = 0

        clf_train_cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
        d_train_cm = np.zeros((2, 2), dtype=np.int64)  # for Discriminator

        with tqdm(
            total=n_train, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch"
        ) as pbar:
            # 短いdataloaderに合わせる
            for clf_l_src_batch, clf_l_trg_batch, d_l_src_batch, d_unl_trg_batch in zip(
                clf_l_src_train_loader, clf_l_trg_train_loader, d_l_src_train_loader, d_unl_trg_train_loader
            ):
                clf_src_imgs, clf_src_labels = clf_l_src_batch["image"], clf_l_src_batch["label"]
                clf_trg_imgs, clf_trg_labels = clf_l_trg_batch["image"], clf_l_trg_batch["label"]
                d_src_imgs = d_l_src_batch["image"]
                d_trg_imgs = d_unl_trg_batch["image"]

                # classification用
                clf_imgs = torch.cat((clf_src_imgs, clf_trg_imgs), 0)
                clf_labels = torch.cat((clf_src_labels, clf_trg_labels), 0)

                clf_imgs = clf_imgs.to(device=device, dtype=torch.float32)  # class-balanced
                clf_labels = clf_labels.to(device=device, dtype=torch.long)

                # Adversarial loss用
                adv_labels = \
                    torch.full((d_trg_imgs.data.size()[0],), D_src_label, dtype=torch.long, device=device)

                # Discriminator用
                D_src_imgs = d_src_imgs.to(device=device, dtype=torch.float32)  # cluster-balanced
                D_trg_imgs = d_trg_imgs.to(device=device, dtype=torch.float32)  # cluster-balanced
                D_src_labels = \
                    torch.full((D_src_imgs.data.size()[0],), D_src_label, dtype=torch.long, device=device)
                D_trg_labels = \
                    torch.full((D_trg_imgs.data.size()[0],), D_trg_label, dtype=torch.long, device=device)

                # ===== train Encoder ===== #
                # don't accumulate grads in D
                for param in netD.parameters():
                    param.requires_grad = False

                optE.zero_grad()

                # train with labeled source-target
                clf_pred = netE(clf_imgs, mode="class")
                clf_loss = criterion(clf_pred, clf_labels)
                clf_loss = 1000 * clf_loss  # 係数(1000)は要調整
                clf_loss.backward()
                EC_loss_value += clf_loss.item()

                # train with unlabeled target
                D_trg_featuremap = netE(D_trg_imgs, mode="feature")
                D_trg_out = netD(D_trg_featuremap)
                A_trg_loss = criterion(D_trg_out, adv_labels)
                A_trg_loss.backward()
                A_loss_value += A_trg_loss.item()

                nn.utils.clip_grad_value_(netE.parameters(), 0.1)
                optE.step()

                #  ====== train Discriminator ====== #
                # bring back requires_grad
                for param in netD.parameters():
                    param.requires_grad = True

                optD.zero_grad()

                # train with source
                D_src_featuremap = netE(D_src_imgs, mode="feature")
                D_src_featuremap = D_src_featuremap.detach()  # stop updating Encoder(CNN)
                D_src_out = netD(D_src_featuremap)
                D_src_loss = criterion(D_src_out, D_src_labels)
                D_src_loss.backward()
                D_loss_value += D_src_loss.item()

                # train with target
                D_trg_featuremap = netE(D_trg_imgs, mode="feature")
                D_trg_featuremap = D_trg_featuremap.detach()  # stop updating Encoder(CNN)
                D_trg_out = netD(D_trg_featuremap)
                D_trg_loss = criterion(D_trg_out, D_trg_labels)
                D_trg_loss.backward()
                D_loss_value += D_trg_loss.item()

                nn.utils.clip_grad_value_(netD.parameters(), 0.1)
                optD.step()

                clf_train_cm, d_train_cm = \
                    eval_net_train(
                        clf_preds=clf_pred,
                        clf_labels=clf_labels,
                        d_src_preds=D_src_out,
                        d_src_labels=D_src_labels,
                        d_trg_preds=D_trg_out,
                        d_trg_labels=D_trg_labels,
                        clf_cm=clf_train_cm,
                        d_cm=d_train_cm,
                    )

                pbar.update(1)

        # calculate averaged training loss
        EC_loss_value /= n_train
        A_loss_value /= n_train
        D_loss_value /= n_train

        # calculate validation loss and confusion matrix
        clf_val_loss, d_val_loss, clf_val_cm, d_val_cm = \
            eval_net_trg_val(netE, netD, val_loader, criterion, device, trg_label=D_trg_label)

        # calculate validation metircs
        val_metrics = eval_metrics(clf_val_cm)

        if stop_cond == "val_loss":
            cond_val = clf_val_loss
        else:
            cond_val = val_metrics[stop_cond]

        best_model_info = update_best_model(cond_val, epoch, best_model_info, mode=mode)
        logging.info("\n EC_Loss(train, epoch): {}".format(EC_loss_value))
        logging.info("\n A_Loss (train, epoch): {}".format(A_loss_value))
        logging.info("\n D_Loss (train, epoch): {}".format(D_loss_value))
        logging.info("\n EC_Loss(valid, epoch): {}".format(clf_val_loss))
        logging.info("\n Acc    (valid, epoch): {}".format(val_metrics['accuracy']))
        logging.info("\n Prec   (valid, epoch): {}".format(val_metrics['precision']))
        logging.info("\n Recall (valid, epoch): {}".format(val_metrics['recall']))
        logging.info("\n mIoU   (valid, epoch): {}".format(val_metrics['mIoU']))

        # calculate unlabeled target loss and confusion matrix (for test)
        clf_unl_trg_loss, d_unl_trg_loss, clf_unl_trg_cm, d_unl_trg_cm = \
            eval_net_trg_val(netE, netD, unl_trg_test_loader, criterion, device, trg_label=D_trg_label)

        if writer is not None:
            # training log
            writer = tensorboard_logging(
                writer=writer,
                epoch=epoch,
                loss_EC=EC_loss_value,
                loss_A=A_loss_value,
                loss_D=D_loss_value,
                l_cm=clf_train_cm,
                unl_cm=None,
                d_cm=d_train_cm,
                classes=classes,
                mode="train"
            )

            # validation log
            writer = tensorboard_logging(
                writer=writer,
                epoch=epoch,
                loss_EC=clf_val_loss,
                loss_A=None,
                loss_D=d_val_loss,
                l_cm=clf_val_cm,
                unl_cm=None,
                d_cm=d_val_cm,
                classes=classes,
                mode="valid"
            )

            # test log (unlabeled target)
            writer = tensorboard_logging(
                writer=writer,
                epoch=epoch,
                loss_EC=clf_unl_trg_loss,
                loss_A=None,
                loss_D=d_unl_trg_loss,
                l_cm=clf_unl_trg_cm,
                unl_cm=None,
                d_cm=d_unl_trg_cm,
                classes=classes,
                mode="test (unl_trg)"
            )

        if best_model_info["epoch"] == epoch:
            torch.save(
                netE.state_dict(),
                checkpoint_dir + f"cv{cv_num}_E_epoch{epoch + 1:03d}.pth",
            )
            torch.save(
                netD.state_dict(),
                checkpoint_dir + f"cv{cv_num}_D_epoch{epoch + 1:03d}.pth",
            )
            logging.info(f"Checkpoint {epoch + 1} saved !")

        if early_stop(cond_val, epoch, best_model_info, patience=patience, mode=mode):
            break

    if writer is not None:
        writer.close()


def main(config_path: str):
    fix_seed(0)

    with open(config_path) as file:
        config = yaml.safe_load(file.read())

    input_shape = tuple(config['main']['shape'])
    transform = {"Resize": True, "HFlip": True, "VFlip": True}

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    # WSIのリストを取得 (target)
    l_trg_wsis = joblib.load(
        config['dataset']['jb_dir']
        + f"{config['main']['trg_facility']}/"
        + "trg_l_wsi.jb"
    )
    trg_valid_wsis = joblib.load(
        config['dataset']['jb_dir']
        + f"{config['main']['trg_facility']}/"
        + "valid_wsi.jb"
    )
    unl_trg_wsis = joblib.load(
        config['dataset']['jb_dir']
        + f"{config['main']['trg_facility']}/"
        + "trg_unl_wsi.jb"
    )

    for cv_num in range(config["main"]["cv"]):
        for l_trg_num, l_trg_selected_wsi in enumerate(l_trg_wsis):
            if l_trg_num != 0:
                continue

            logging.info(f"== CV{cv_num}: {l_trg_selected_wsi} ==")
            writer = SummaryWriter(
                log_dir=(
                    (
                        f"{config['main']['result_dir']}logs/{config['main']['prefix']}_{config['main']['src_facility']}_"
                        + f"{l_trg_selected_wsi}_{config['main']['model']}_batch{config['main']['batch_size']}_"
                        + f"shape{config['main']['shape']}_cl{config['main']['classes']}_cv{cv_num}"
                    )
                )
            )

            # 事前学習済みの重みを読み込み
            if config['main']['load_pretrained_weight_E']:
                weight_E_path = (
                    config['main']['pretrained_weight_E_dir']
                    + config['main']['pretrained_weight_E_names'][cv_num]
                )
                if os.path.exists(weight_E_path) is False:
                    logging.info(f"weight_E_path: {weight_E_path} does not exist")
                    weight_E_path = None
            else:
                weight_E_path = None

            # モデルを取得
            netE = Encoder(
                encoder_name=config['main']['model'],
                num_classes=len(config['main']['classes']),
                pretrained=True, weight_path=weight_E_path, device=device
            ).to(device)

            netD = Discriminator(
                encoder_name=config['main']['model']
            ).to(device)

            # WSIのリストを取得 (source)
            l_src_wsis = joblib.load(
                config['dataset']['jb_dir']
                + f"{config['main']['src_facility']}/"
                + f"cv{cv_num}_"
                + f"train_{config['main']['src_facility']}_wsi.jb"
            )

            dataset = WSIDataset_ST1_ADA2_ValT(
                l_src_wsis=l_src_wsis,
                l_trg_wsi=l_trg_selected_wsi,
                unl_trg_wsis=unl_trg_wsis,
                trg_valid_wsis=trg_valid_wsis,
                src_imgs_dir=config['dataset']['src_imgs_dir'],
                trg_imgs_dir=config['dataset']['trg_imgs_dir'],
                classes=config['main']['classes'],
                shape=input_shape,
                transform=transform,
                batch_size=config['main']['batch_size'],
                cluster_num=config['main']['cluster_num'],
                weight_path=weight_E_path,
                balance_domain=config['main']['balance_domain'],
            )

            clf_l_src_data, clf_l_trg_data, d_src_data, d_trg_data, valid_data, test_data = dataset.get()

            logging.info(
                f"""Starting training:
                Classes:            {config['main']['classes']}
                Epochs:             {config['main']['epochs']}
                Batch size:         {config['main']['batch_size']}
                Model:              {config['main']['model']}
                Optim:              {config['main']['optim']}
                Transform:          {json.dumps(transform)}
                Train clf (l_src):  {len(clf_l_src_data)}
                Train clf (l_trg):  {len(clf_l_trg_data)}
                Train d (l_src):    {len(clf_l_src_data)}
                Train d (unl_trg):  {len(clf_l_trg_data)}
                Validation:         {len(valid_data)}
                Test (unl_trg):     {len(test_data)}
                Patience:           {config['main']['patience']}
                StopCond:           {config['main']['stop_cond']}
                Device:             {device.type}
                Images Shape:       {input_shape}
                Source Facility:    {config['main']['src_facility']}
                Target Facility:    {config['main']['trg_facility']}
                Selected WSI:       {l_trg_selected_wsi}
            """
            )

            checkpoint_dir = (
                f"{config['main']['result_dir']}checkpoints/"
                + f"{config['main']['prefix']}_ADA_{config['main']['src_facility']}_{l_trg_selected_wsi}_{config['main']['classes']}/")
            try:
                os.mkdir(checkpoint_dir)
                logging.info("Created checkpoint directory")
            except OSError:
                pass

            try:
                train_net(
                    netE=netE,
                    netD=netD,
                    clf_l_src_train_data=clf_l_src_data,
                    clf_l_trg_train_data=clf_l_trg_data,
                    d_l_src_train_data=d_src_data,
                    d_unl_trg_train_data=d_trg_data,
                    valid_data=valid_data,
                    unl_trg_test_data=test_data,
                    device=device,
                    epochs=config['main']['epochs'],
                    batch_size=config['main']['batch_size'],
                    classes=config['main']['classes'],
                    checkpoint_dir=checkpoint_dir,
                    writer=writer,
                    patience=config['main']['patience'],
                    stop_cond=config['main']['stop_cond'],
                    cv_num=cv_num,
                )

            except KeyboardInterrupt:
                torch.save(
                    netE.state_dict(),
                    config['main']['result_dir'] + f"cv{cv_num}_E_INTERRUPTED.pth",
                )
                torch.save(
                    netD.state_dict(),
                    config['main']['result_dir'] + f"cv{cv_num}_D_INTERRUPTED.pth",
                )
                logging.info("Saved interrupt")
                try:
                    sys.exit(0)
                except SystemExit:
                    os._exit(0)


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    config_path = "../ST_ADA2/config_st-ada2_cl[0, 1, 2]_valt3.yaml"
    # config_path = "./ST_ADA2/config_st-ada2_cl[0, 1, 2]_valt3.yaml"
    main(config_path=config_path)
