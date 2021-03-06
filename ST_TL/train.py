import os
import sys
import yaml
import joblib
import json
import logging
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from ST_TL.dataset import WSIDatasetST1_TL_ValT
from S.eval import eval_net, plot_confusion_matrix, convert_plt2nd, eval_metrics
from S.util import fix_seed, ImbalancedDatasetSampler, select_optim
from S.model import build_model
from S.train import early_stop, update_best_model


def train_net(
    net,
    trg_train_data,
    valid_data,
    device,
    epochs=5,
    batch_size=16,
    optim_name="Adam",
    classes=[0, 1, 2],
    checkpoint_dir="checkpoints/",
    writer=None,
    patience=5,
    stop_cond="mIoU",
    cv_num=0,
):

    logging.info(f"samples num in each iteration: {len(trg_train_data)}")

    trg_train_loader = DataLoader(
        trg_train_data,
        sampler=ImbalancedDatasetSampler(trg_train_data),
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

    n_train = len(trg_train_loader)

    optimizer = select_optim(optim_name, net.parameters())
    criterion = nn.CrossEntropyLoss()

    if stop_cond == "val_loss":
        mode = "min"
    else:
        mode = "max"

    if mode == "min":
        best_model_info = {"epoch": 0, "val": float("inf")}
    elif mode == "max":
        best_model_info = {"epoch": 0, "val": float("-inf")}

    for epoch in range(epochs):
        net.train()

        # netのClassifierは勾配計算をしない
        for param in net.fc.parameters():
            param.requires_grad = False

        epoch_loss = 0
        with tqdm(
            total=n_train, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch"
        ) as pbar:
            # 短いdataloaderに合わせる
            for trg_batch in trg_train_loader:
                trg_imgs, trg_labels = trg_batch["image"], trg_batch["label"]

                imgs = trg_imgs.to(device=device, dtype=torch.float32)
                labels = trg_labels.to(device=device, dtype=torch.long)

                preds = net(imgs)

                loss = criterion(preds, labels)

                epoch_loss += loss.item()
                pbar.set_postfix(**{"loss (batch)": loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                pbar.update(1)

        # calculate validation loss and confusion matrix
        val_loss, cm = eval_net(net, val_loader, criterion, device)

        # calculate validation metircs
        val_metrics = eval_metrics(cm)

        if stop_cond == "val_loss":
            cond_val = val_loss
        else:
            cond_val = val_metrics[stop_cond]

        best_model_info = update_best_model(cond_val, epoch, best_model_info, mode=mode)
        logging.info("\n Loss   (train, epoch): {}".format(epoch_loss))
        logging.info("\n Loss   (valid, batch): {}".format(val_loss))
        logging.info("\n Acc    (valid, epoch): {}".format(val_metrics['accuracy']))
        logging.info("\n Prec   (valid, epoch): {}".format(val_metrics['precision']))
        logging.info("\n Recall (valid, epoch): {}".format(val_metrics['recall']))
        logging.info("\n mIoU   (valid, epoch): {}".format(val_metrics['mIoU']))

        if writer is not None:
            # upload loss (train) and learning_rate to tensorboard
            writer.add_scalar("Loss/train", epoch_loss, epoch)

            # upload confusion_matrix (validation) to tensorboard
            cm_plt = plot_confusion_matrix(cm, classes, normalize=True)
            cm_nd = convert_plt2nd(cm_plt)
            writer.add_image(
                "confusion_matrix/valid", cm_nd, global_step=epoch, dataformats="HWC"
            )
            plt.clf()
            plt.close()

            # upload not-normed confusion_matrix (validation) to tensorboard
            cm_plt = plot_confusion_matrix(cm, classes, normalize=False)
            cm_nd = convert_plt2nd(cm_plt)
            writer.add_image(
                "confusion_matrix_nn/valid", cm_nd, global_step=epoch, dataformats="HWC"
            )
            plt.clf()
            plt.close()

            # upload loss & score (validation) to tensorboard
            writer.add_scalar("Loss/valid", val_loss, epoch)
            writer.add_scalar("mIoU/valid", val_metrics['mIoU'], epoch)
            writer.add_scalar("Accuracy/valid", val_metrics['accuracy'], epoch)
            writer.add_scalar("Precision/valid", val_metrics['precision'], epoch)
            writer.add_scalar("Recall/valid", val_metrics['recall'], epoch)
            writer.add_scalar("F1/valid", val_metrics['f1'], epoch)

        if best_model_info["epoch"] == epoch:
            torch.save(
                net.state_dict(),
                checkpoint_dir + f"cv{cv_num}_epoch{epoch + 1}.pth",
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
    trg_train_wsis = joblib.load(
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

    for cv_num in range(config["main"]["cv"]):
        for trg_selected_wsi in trg_train_wsis:
            logging.info(f"== CV{cv_num}: {trg_selected_wsi} ==")
            writer = SummaryWriter(
                log_dir=(
                    (
                        f"{config['main']['result_dir']}logs/{config['main']['prefix']}_{config['main']['src_facility']}_"
                        + f"{trg_selected_wsi}_{config['main']['model']}_batch{config['main']['batch_size']}_"
                        + f"shape{config['main']['shape']}_cl{config['main']['classes']}_cv{cv_num}"
                    )
                )
            )

            # モデルを取得
            net = build_model(
                config['main']['model'], num_classes=len(config['main']['classes'])
            )
            net.to(device=device)
            # 事前学習済みの重みを読み込み
            if config['main']['load_pretrained_weight']:
                weight_path = (
                    config['main']['pretrained_weight_dir']
                    + config['main']['pretrained_weight_names'][cv_num]
                )
                net.load_state_dict(torch.load(weight_path, map_location=device))
                print(f"load_weight: {weight_path}")

            dataset = WSIDatasetST1_TL_ValT(
                trg_train_wsi=trg_selected_wsi,
                trg_valid_wsis=trg_valid_wsis,
                trg_test_wsis=trg_test_wsis,
                trg_imgs_dir=config['dataset']['trg_imgs_dir'],
                classes=config['main']['classes'],
                shape=input_shape,
                transform=transform,
            )

            trg_train_data, valid_data, test_data = dataset.get()

            logging.info(
                f"""Starting training:
                Classes:           {config['main']['classes']}
                Epochs:            {config['main']['epochs']}
                Batch size:        {config['main']['batch_size']}
                Model:             {config['main']['model']}
                Optim:             {config['main']['optim']}
                Transform:         {json.dumps(transform)}
                Training size(trg):{len(trg_train_data)}
                Validation size:   {len(valid_data)}
                Patience:          {config['main']['patience']}
                StopCond:          {config['main']['stop_cond']}
                Device:            {device.type}
                Images Shape:      {input_shape}
                Source Facility:   {config['main']['src_facility']}
                Target Facility:   {config['main']['trg_facility']}
                Selected WSI:      {trg_selected_wsi}
            """
            )

            checkpoint_dir = (
                f"{config['main']['result_dir']}checkpoints/"
                + f"{config['main']['prefix']}_{config['main']['src_facility']}_{trg_selected_wsi}_{config['main']['classes']}/")
            try:
                os.mkdir(checkpoint_dir)
                logging.info("Created checkpoint directory")
            except OSError:
                pass

            try:
                train_net(
                    net=net,
                    trg_train_data=trg_train_data,
                    valid_data=valid_data,
                    epochs=config['main']['epochs'],
                    batch_size=config['main']['batch_size'],
                    device=device,
                    classes=config['main']['classes'],
                    checkpoint_dir=checkpoint_dir,
                    writer=writer,
                    patience=config['main']['patience'],
                    stop_cond=config['main']['stop_cond'],
                    cv_num=cv_num,
                )
            except KeyboardInterrupt:
                torch.save(
                    net.state_dict(),
                    config['main']['result_dir'] + f"cv{cv_num}_INTERRUPTED.pth",
                )
                logging.info("Saved interrupt")
                try:
                    sys.exit(0)
                except SystemExit:
                    os._exit(0)


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    config_path = "../ST_TL/config_st_tl_cl[0, 1, 2]_valt3.yaml"
    # config_path = "./ST_TL/config_st_tl_cl[0, 1, 2]_valt3.yaml"
    main(config_path=config_path)
