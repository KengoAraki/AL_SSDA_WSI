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
from src_s_t1.dataset import WSIDatasetST1
from src_s_t1.eval import eval_net, plot_confusion_matrix, convert_plt2nd, eval_metrics
from src_s_t1.util import fix_seed, ImbalancedDatasetSampler, select_optim
from src_s_t1.model import build_model


def train_net(
    net,
    train_data,
    valid_data,
    device,
    epochs=5,
    batch_size=4,
    optim_name="Adam",
    save_cp=True,
    classes=[[0, 1, 2], 3],
    checkpoint_dir="checkpoints/",
    writer=None,
    patience=5,
    stop_cond="recall",
    mode="max",
    cv_num=0,
):

    n_train = len(train_data)

    train_loader = DataLoader(
        train_data,
        sampler=ImbalancedDatasetSampler(train_data),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        valid_data,
        sampler=None,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )

    optimizer = select_optim(optim_name, net.parameters())

    if len(classes) > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    if mode == "min":
        best_model_info = {"epoch": 0, "val": float("inf")}
    elif mode == "max":
        best_model_info = {"epoch": 0, "val": float("-inf")}

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        counter = 1
        with tqdm(
            total=n_train, desc=f"Epoch {epoch + 1}/{epochs}", unit="img"
        ) as pbar:
            for batch in train_loader:
                imgs = batch["image"]
                labels = batch["label"]
                # names = batch['name']

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
        cond_val = val_metrics[stop_cond]

        best_model_info = update_best_model(cond_val, epoch, best_model_info, mode=mode)
        logging.info("\n Loss (train, epoch): {}".format(epoch_loss))
        logging.info("\n Loss (valid, batch): {}".format(val_loss))
        logging.info("\n mIoU (valid, epoch): {}".format(val_metrics["mIoU"]))

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
            writer.add_scalar("mIoU/valid", val_metrics["mIoU"], epoch)
            writer.add_scalar("Accuracy/valid", val_metrics["accuracy"], epoch)
            writer.add_scalar("Precision/valid", val_metrics["precision"], epoch)
            writer.add_scalar("Recall/valid", val_metrics["recall"], epoch)
            writer.add_scalar("F1/valid", val_metrics["f1"], epoch)

            # # upload images (train) and their results to tensorboard
            # imgs_list = []
            # for i in range(5):
            #     pred_label = int(preds[i].argmax(dim=0).cpu().item())
            #     true_label = int(labels[i].cpu().item())
            #     img = imgs[i].cpu().numpy()
            #     img = put_label2img(
            #         img, pred_label, true_label, name=None, is_transpose=True, is_mul=True)
            #     imgs_list.append(img)
            # imgs = np.array(imgs_list)
            # # NHWC -> NCHW (for torch tensor)
            # imgs = torch.from_numpy(np.transpose(imgs, [0, 3, 1, 2]))
            # nrow = len(imgs_list) if 5 > len(imgs_list) else 5
            # imgs_grid = torchvision.utils.make_grid(imgs, nrow=nrow, padding=5)
            # writer.add_image(
            #     'results/imgs_train',
            #     imgs_grid,
            #     global_step=epoch
            # )

        counter += 1

        if save_cp:
            try:
                os.mkdir(checkpoint_dir)
                logging.info("Created checkpoint directory")
            except OSError:
                pass

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


def update_best_model(val, epoch, best_model_info, mode="max"):
    if mode == "min":
        if val < best_model_info["val"]:
            best_model_info["val"] = val
            best_model_info["epoch"] = epoch
            print(
                f"[Best Model] epoch: {best_model_info['epoch']}, \
                val: {best_model_info['val']}"
            )
    elif mode == "max":
        if val > best_model_info["val"]:
            best_model_info["val"] = val
            best_model_info["epoch"] = epoch
            print(
                f"[Best Model] epoch: {best_model_info['epoch']}, \
                val: {best_model_info['val']}"
            )
    else:
        sys.exit("select mode max or min")
    return best_model_info


def early_stop(val, epoch, best_model_info, patience=5, mode="max"):
    terminate = False
    if (epoch - best_model_info["epoch"]) == patience:
        if mode == "min":
            if val >= best_model_info["val"]:
                terminate = True
        elif mode == "max":
            if val <= best_model_info["val"]:
                terminate = True
        else:
            sys.exit("select mode max or min")
    return terminate


def main():
    fix_seed(0)
    # config_path = "./config/config_s_t1_cl[0, 1, 2].yaml"
    config_path = "../config/config_s_t1_cl[0, 1, 2].yaml"

    with open(config_path) as file:
        config = yaml.safe_load(file.read())

    input_shape = tuple(config["main"]["shape"])
    transform = {"Resize": True, "HFlip": True, "VFlip": True}

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    for cv_num in range(config["main"]["cv"]):
        for trg_selected_wsi in config["main"]["trg_selected_wsis"]:
            logging.info(f"== CV{cv_num}: {trg_selected_wsi} ==")
            writer = SummaryWriter(
                log_dir=(
                    (
                        f"{config['main']['result_dir']}logs/{config['main']['src_facility']}_"
                        + f"{trg_selected_wsi}_{config['main']['model']}_{config['main']['optim']}_"
                        + f"batch{config['main']['batch_size']}_shape{config['main']['shape']}_cl{config['main']['classes']}_cv{cv_num}"
                    )
                )
            )

            # モデルを取得
            net = build_model(
                config["main"]["model"], num_classes=len(config["main"]["classes"])
            )

            net.to(device=device)
            # 事前学習済みの重みを読み込み
            if config["main"]["load_pretrained_weight"]:
                weight_dir = (
                    config["main"]["pretrained_weight_dir"]
                    + f"{config['main']['src_facility']}_"
                    + f"{config['main']['classes']}/"
                )
                weight_path = (
                    weight_dir + config["main"]["pretrained_weight_names"][cv_num]
                )
                net.load_state_dict(torch.load(weight_path, map_location=device))
                print(f"load_weight: {weight_path}")

            # WSIのリストを取得 (source)
            src_train_wsis = joblib.load(
                config["main"]["jb_dir"]
                + f"{config['main']['src_facility']}/"
                + f"cv{cv_num}_"
                + f"train_{config['main']['src_facility']}_wsi.jb"
            )
            src_valid_wsis = joblib.load(
                config["main"]["jb_dir"]
                + f"{config['main']['src_facility']}/"
                + f"cv{cv_num}_"
                + f"valid_{config['main']['src_facility']}_wsi.jb"
            )
            src_test_wsis = joblib.load(
                config["main"]["jb_dir"]
                + f"{config['main']['src_facility']}/"
                + f"cv{cv_num}_"
                + f"test_{config['main']['src_facility']}_wsi.jb"
            )

            # WSIのリストを取得 (target)
            trg_train_wsis = joblib.load(
                config["main"]["jb_dir"]
                + f"{config['main']['trg_facility']}/"
                + f"cv{cv_num}_"
                + f"train_{config['main']['trg_facility']}_wsi.jb"
            )
            trg_valid_wsis = joblib.load(
                config["main"]["jb_dir"]
                + f"{config['main']['trg_facility']}/"
                + f"cv{cv_num}_"
                + f"valid_{config['main']['trg_facility']}_wsi.jb"
            )
            trg_test_wsis = joblib.load(
                config["main"]["jb_dir"]
                + f"{config['main']['trg_facility']}/"
                + f"cv{cv_num}_"
                + f"test_{config['main']['trg_facility']}_wsi.jb"
            )
            trg_all_wsis = trg_train_wsis + trg_valid_wsis + trg_test_wsis

            dataset = WSIDatasetST1(
                src_train_wsis=src_train_wsis,
                src_valid_wsis=src_valid_wsis,
                src_test_wsis=src_test_wsis,
                trg_all_wsis=trg_all_wsis,
                trg_selected_wsi=trg_selected_wsi,
                src_imgs_dir=config["dataset"]["src_imgs_dir"],
                trg_imgs_dir=config["dataset"]["trg_imgs_dir"],
                classes=config["main"]["classes"],
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
                Source Facility:   {config['main']['src_facility']}
                Target Facility:   {config['main']['trg_facility']}
                Selected WSI:      {trg_selected_wsi}
            """
            )

            try:
                train_net(
                    net=net,
                    train_data=train_data,
                    valid_data=valid_data,
                    epochs=config["main"]["epochs"],
                    batch_size=config["main"]["batch_size"],
                    device=device,
                    classes=config["main"]["classes"],
                    checkpoint_dir=f"{config['main']['result_dir']}checkpoints/{config['main']['src_facility']}_{trg_selected_wsi}_{config['main']['classes']}/",
                    writer=writer,
                    patience=config["main"]["patience"],
                    stop_cond=config["main"]["stop_cond"],
                    mode="max",
                    cv_num=cv_num,
                )
            except KeyboardInterrupt:
                torch.save(
                    net.state_dict(),
                    config["main"]["result_dir"] + f"cv{cv_num}_INTERRUPTED.pth",
                )
                logging.info("Saved interrupt")
                try:
                    sys.exit(0)
                except SystemExit:
                    os._exit(0)


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    main()
