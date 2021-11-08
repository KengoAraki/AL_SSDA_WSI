import os
import sys
import yaml
import joblib
import json
import logging
import torch
import torch.nn as nn
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from ST.dataset import WSIDataset
from ST.eval import eval_net, eval_metrics
from ST.util import fix_seed, ImbalancedDatasetSampler, select_optim
from ST.model import build_model


def train_net(
    net,
    train_data,
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

    n_train = len(train_data)

    train_loader = DataLoader(
        train_data,
        sampler=ImbalancedDatasetSampler(train_data),
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

    optimizer = select_optim(optim_name, net.parameters())

    if len(classes) > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

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

        epoch_loss = 0
        with tqdm(
            total=n_train, desc=f"Epoch {epoch + 1}/{epochs}", unit="img"
        ) as pbar:
            # 短いdataloaderに合わせる
            for batch in train_loader:
                imgs = batch['image']
                labels = batch['label']

                imgs = imgs.to(device=device, dtype=torch.float32)
                labels = labels.to(device=device, dtype=torch.long)

                preds = net(imgs)

                loss = criterion(preds, labels)

                epoch_loss += loss.item()
                pbar.set_postfix(**{"loss (batch)": loss.item()})
                pbar.update(imgs.shape[0])

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

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

            # # upload confusion_matrix (validation) to tensorboard
            # cm_plt = plot_confusion_matrix(cm, classes, normalize=True)
            # cm_nd = convert_plt2nd(cm_plt)
            # writer.add_image(
            #     "confusion_matrix/valid", cm_nd, global_step=epoch, dataformats="HWC"
            # )
            # plt.clf()
            # plt.close()

            # # upload not-normed confusion_matrix (validation) to tensorboard
            # cm_plt = plot_confusion_matrix(cm, classes, normalize=False)
            # cm_nd = convert_plt2nd(cm_plt)
            # writer.add_image(
            #     "confusion_matrix_nn/valid", cm_nd, global_step=epoch, dataformats="HWC"
            # )
            # plt.clf()
            # plt.close()

            # upload loss & score (validation) to tensorboard
            writer.add_scalar("Loss/valid", val_loss, epoch)
            writer.add_scalar("mIoU/valid", val_metrics['mIoU'], epoch)
            writer.add_scalar("Accuracy/valid", val_metrics['accuracy'], epoch)
            writer.add_scalar("Precision/valid", val_metrics['precision'], epoch)
            writer.add_scalar("Recall/valid", val_metrics['recall'], epoch)
            writer.add_scalar("F1/valid", val_metrics['f1'], epoch)

        if best_model_info['epoch'] == epoch:
            torch.save(
                net.state_dict(),
                checkpoint_dir + f"cv{cv_num}_epoch{epoch + 1}.pth",
            )
            logging.info(f"Checkpoint {epoch + 1} saved !")

        if early_stop(cond_val, epoch, best_model_info, patience=patience, mode=mode):
            break

    if writer is not None:
        writer.close()


def update_best_model(val, epoch, best_model_info, mode="max"):
    if mode == "min":
        if val < best_model_info['val']:
            best_model_info['val'] = val
            best_model_info['epoch'] = epoch
            print(
                f"[Best Model] epoch: {best_model_info['epoch']}, \
                val: {best_model_info['val']}"
            )
    elif mode == "max":
        if val > best_model_info['val']:
            best_model_info['val'] = val
            best_model_info['epoch'] = epoch
            print(
                f"[Best Model] epoch: {best_model_info['epoch']}, \
                val: {best_model_info['val']}"
            )
    else:
        sys.exit("select mode max or min")
    return best_model_info


def early_stop(val, epoch, best_model_info, patience=5, mode="max"):
    terminate = False
    if (epoch - best_model_info['epoch']) == patience:
        if mode == "min":
            if val >= best_model_info['val']:
                terminate = True
        elif mode == "max":
            if val <= best_model_info['val']:
                terminate = True
        else:
            sys.exit("select mode max or min")
    return terminate


# sourceのみで学習
def main():
    fix_seed(0)
    config_path = "./S/config_s_cl[0, 1, 2].yaml"

    with open(config_path) as file:
        config = yaml.safe_load(file.read())

    input_shape = tuple(config['main']['shape'])
    transform = {"Resize": True, "HFlip": True, "VFlip": True}

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    for cv_num in range(config['main']['cv']):
        logging.info(f"== CV{cv_num} ==")
        writer = SummaryWriter(
            log_dir=(
                (
                    f"{config['main']['result_dir']}logs/s_{config['main']['facility']}_"
                    + f"{config['main']['model']}_batch{config['main']['batch_size']}_"
                    + f"shape{config['main']['shape']}_cl{config['main']['classes']}_cv{cv_num}"
                )
            )
        )

        # モデルを取得
        net = build_model(
            config['main']['model'], num_classes=len(config['main']['classes'])
        )
        net.to(device=device)

        # WSIのリストを取得 (source)
        train_wsis = joblib.load(
            config['main']['jb_dir']
            + f"{config['main']['facility']}/"
            + f"cv{cv_num}_"
            + f"train_{config['main']['facility']}_wsi.jb"
        )
        valid_wsis = joblib.load(
            config['main']['jb_dir']
            + f"{config['main']['facility']}/"
            + f"cv{cv_num}_"
            + f"valid_{config['main']['facility']}_wsi.jb"
        )
        test_wsis = joblib.load(
            config['main']['jb_dir']
            + f"{config['main']['facility']}/"
            + f"cv{cv_num}_"
            + f"test_{config['main']['facility']}_wsi.jb"
        )

        dataset = WSIDataset(
            train_wsis=train_wsis,
            valid_wsis=valid_wsis,
            test_wsis=test_wsis,
            imgs_dir=config['dataset']['imgs_dir'],
            classes=config['main']['classes'],
            shape=input_shape,
            transform=transform,
        )

        train_data, valid_data, test_data = dataset.get()
        train_wsi, valid_wsi, test_wsi = dataset.get_wsi_split()

        logging.info(
            f"""Starting training:
            Classes:           {config['main']['classes']}
            Epochs:            {config['main']['epochs']}
            Batch size:        {config['main']['batch_size']}
            Model:             {config['main']['model']}
            Optim:             {config['main']['optim']}
            Transform:         {json.dumps(transform)}
            Training size:     {len(train_data)}
            Validation size:   {len(valid_data)}
            Patience:          {config['main']['patience']}
            StopCond:          {config['main']['stop_cond']}
            Device:            {device.type}
            Images Shape:      {input_shape}
            Source Facility:   {config['main']['facility']}
        """
        )

        checkpoint_dir = (
            f"{config['main']['result_dir']}checkpoints/"
            + f"s_{config['main']['facility']}_{config['main']['classes']}/")
        try:
            os.mkdir(checkpoint_dir)
            logging.info("Created checkpoint directory")
        except OSError:
            pass

        try:
            train_net(
                net=net,
                train_data=train_data,
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

    main()
