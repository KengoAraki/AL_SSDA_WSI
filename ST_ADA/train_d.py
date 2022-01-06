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
import copy

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from S.train import update_best_model, early_stop
from S.eval import eval_metrics
from S.util import fix_seed, ImbalancedDatasetSampler, select_optim
from ST_ADA.model import Encoder, Discriminator, Discriminator2
from ST_ADA.dataset import WSIDataset_ST1_ADA_ValT_d
from ST_ADA.eval import eval_net_train_d, eval_net_trg_val_d, tensorboard_logging


def train_net(
    netE,
    netD,
    l_src_train_data,
    l_trg_train_data,
    unl_trg_train_data,
    src_valid_data,
    trg_valid_data,
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
    l_src_train_data: labeled source train datasaet (torch.utils.data.Dataset),
    l_trg_train_data: labeled target train datasaet (torch.utils.data.Dataset),
    unl_trg_train_data: unlabeled target train dataset (torch.utils.data.Dataset),
    src_valid_data: source validation dataset (torch.utils.data.Dataset),
    trg_valid_data: target validation dataset (torch.utils.data.Dataset),
    """

    l_src_train_loader = DataLoader(
        l_src_train_data,
        sampler=ImbalancedDatasetSampler(l_src_train_data),
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    l_trg_train_loader = DataLoader(
        l_trg_train_data,
        sampler=ImbalancedDatasetSampler(l_trg_train_data),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    unl_trg_train_loader = DataLoader(
        unl_trg_train_data,
        sampler=ImbalancedDatasetSampler(unl_trg_train_data),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    src_val_loader = DataLoader(
        src_valid_data,
        sampler=None,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    trg_val_loader = DataLoader(
        trg_valid_data,
        sampler=None,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

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

    n_train = min(len(l_src_train_loader), len(l_trg_train_loader), len(unl_trg_train_loader))

    for epoch in range(epochs):
        netE.train()
        netD.train()

        D_loss_value = 0

        d_train_cm = np.zeros((2, 2), dtype=np.int64)  # for Discriminator

        with tqdm(
            total=n_train, desc=f"Epoch {epoch + 1}/{epochs}", unit="img"
        ) as pbar:
            # 短いdataloaderに合わせる
            for l_src_batch, l_trg_batch, unl_trg_batch in zip(
                l_src_train_loader, l_trg_train_loader, unl_trg_train_loader
            ):
                # unl_batchのbatchサイズはl_batchの2倍
                l_src_imgs = l_src_batch["image"]
                l_trg_imgs = l_trg_batch["image"]
                unl_trg_imgs = unl_trg_batch["image"]

                # Discriminator用
                D_src_imgs = l_src_imgs.to(device=device, dtype=torch.float32)
                D_trg_imgs = torch.cat((l_trg_imgs, unl_trg_imgs), 0)
                D_trg_imgs = D_trg_imgs.to(device=device, dtype=torch.float32)
                D_src_labels = \
                    torch.full((l_src_imgs.data.size()[0],), D_src_label, dtype=torch.long, device=device)
                D_trg_labels = \
                    torch.full((D_trg_imgs.data.size()[0],), D_trg_label, dtype=torch.long, device=device)

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

                d_train_cm = \
                    eval_net_train_d(
                        d_src_preds=D_src_out,
                        d_src_labels=D_src_labels,
                        d_trg_preds=D_trg_out,
                        d_trg_labels=D_trg_labels,
                        d_cm=d_train_cm,
                    )

                pbar.update(1)

        # calculate averaged training loss
        D_loss_value /= n_train

        # calculate validation loss and confusion matrix
        d_val_loss, d_val_cm = \
            eval_net_trg_val_d(netE, netD, src_val_loader, trg_val_loader, criterion, device, src_label=D_src_label, trg_label=D_trg_label)

        # calculate validation metircs
        val_metrics = eval_metrics(d_val_cm)

        if stop_cond == "val_loss":
            cond_val = d_val_loss
        else:
            cond_val = val_metrics[stop_cond]

        best_model_info = update_best_model(cond_val, epoch, best_model_info, mode=mode)
        logging.info("\n D_Loss (train, epoch): {}".format(D_loss_value))
        logging.info("\n D_Loss (valid, epoch): {}".format(d_val_loss))
        logging.info("\n Acc    (valid, epoch): {}".format(val_metrics['accuracy']))
        logging.info("\n Prec   (valid, epoch): {}".format(val_metrics['precision']))
        logging.info("\n Recall (valid, epoch): {}".format(val_metrics['recall']))
        logging.info("\n mIoU   (valid, epoch): {}".format(val_metrics['mIoU']))

        if writer is not None:
            # training log
            writer = tensorboard_logging(
                writer=writer,
                epoch=epoch,
                loss_D=D_loss_value,
                d_cm=d_train_cm,
                classes=classes,
                mode="train"
            )

            # validation log
            writer = tensorboard_logging(
                writer=writer,
                epoch=epoch,
                loss_D=d_val_loss,
                d_cm=d_val_cm,
                classes=classes,
                mode="valid"
            )

        if best_model_info["epoch"] == epoch:
            torch.save(
                netD.state_dict(),
                checkpoint_dir + f"cv{cv_num}_pretrainedD_epoch{epoch + 1:03d}.pth",
            )
            logging.info(f"Checkpoint {epoch + 1} saved !")

        if early_stop(cond_val, epoch, best_model_info, patience=patience, mode=mode):
            break

    if writer is not None:
        writer.close()


# source + 1枚のtargetで学習
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
    l_trg_train_wsis = joblib.load(
        config['dataset']['jb_dir']
        + f"{config['main']['trg_facility']}/"
        + "trg_l_wsi.jb"
    )
    trg_valid_wsis = joblib.load(
        config['dataset']['jb_dir']
        + f"{config['main']['trg_facility']}/"
        + "valid_wsi.jb"
    )
    trg_test_wsis = joblib.load(
        config['dataset']['jb_dir']
        + f"{config['main']['trg_facility']}/"
        + "trg_unl_wsi.jb"
    )

    unl_trg_train_wsis = copy.copy(trg_test_wsis)

    for cv_num in range(config["main"]["cv"]):
        for l_trg_num, l_trg_selected_wsi in enumerate(l_trg_train_wsis):
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

            # netD = Discriminator(
            #     encoder_name=config['main']['model']
            # ).to(device)

            netD = Discriminator2(
                encoder_name=config['main']['model']
            ).to(device)

            # WSIのリストを取得 (source)
            l_src_train_wsis = joblib.load(
                config['dataset']['jb_dir']
                + f"{config['main']['src_facility']}/"
                + f"cv{cv_num}_"
                + f"train_{config['main']['src_facility']}_wsi.jb"
            )
            src_valid_wsis = joblib.load(
                config['dataset']['jb_dir']
                + f"{config['main']['src_facility']}/"
                + f"cv{cv_num}_"
                + f"valid_{config['main']['src_facility']}_wsi.jb"
            )

            dataset = WSIDataset_ST1_ADA_ValT_d(
                l_src_train_wsis=l_src_train_wsis,
                l_trg_train_wsi=l_trg_selected_wsi,
                unl_trg_train_wsis=unl_trg_train_wsis,
                src_valid_wsis=src_valid_wsis,
                trg_valid_wsis=trg_valid_wsis,
                trg_test_wsis=trg_test_wsis,
                src_imgs_dir=config['dataset']['src_imgs_dir'],
                trg_imgs_dir=config['dataset']['trg_imgs_dir'],
                classes=config['main']['classes'],
                shape=input_shape,
                transform=transform,
                balance_domain=config['main']['balance_domain'],
            )

            l_src_train_data, l_trg_train_data, unl_trg_train_data, src_valid_data, trg_valid_data, test_data = dataset.get()

            logging.info(
                f"""Starting training:
                Classes:            {config['main']['classes']}
                Epochs:             {config['main']['epochs']}
                Batch size:         {config['main']['batch_size']}
                Model:              {config['main']['model']}
                Optim:              {config['main']['optim']}
                Transform:          {json.dumps(transform)}
                Train size(l_src):  {len(l_src_train_data)}
                Train size(l_trg):  {len(l_trg_train_data)}
                Train size(unl_trg):{len(unl_trg_train_data)}
                Valid size(src):    {len(src_valid_data)}
                Valid size(trg):    {len(trg_valid_data)}
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
            if os.path.exists(f"{config['main']['result_dir']}checkpoints/") is False:
                os.mkdir(f"{config['main']['result_dir']}checkpoints/")

            try:
                os.mkdir(checkpoint_dir)
                logging.info("Created checkpoint directory")
            except OSError:
                pass

            try:
                train_net(
                    netE=netE,
                    netD=netD,
                    l_src_train_data=l_src_train_data,
                    l_trg_train_data=l_trg_train_data,
                    unl_trg_train_data=unl_trg_train_data,
                    src_valid_data=src_valid_data,
                    trg_valid_data=trg_valid_data,
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
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    config_path = "../ST_ADA/config_st-ada_pretrainedD_cl[0, 1, 2]_valt3.yaml"
    # config_path = "./ST_ADA/config_st-ada_pretrainedD_cl[0, 1, 2]_valt3.yaml"
    main(config_path=config_path)
