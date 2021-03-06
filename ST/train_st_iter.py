import os
import sys
import yaml
import joblib
import json
import logging
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from ST.dataset import WSIDatasetST1_ValT
from S.eval import eval_net_iter, eval_metrics
from S.util import fix_seed, ImbalancedDatasetSampler, select_optim
from S.model import build_model


def train_net(
    net,
    src_train_data,
    trg_train_data,
    valid_data,
    device,
    steps=100,
    batch_size=16,
    optim_name="Adam",
    classes=[0, 1, 2],
    checkpoint_dir="checkpoints/",
    writer=None,
    patience=10,
    stop_cond="mIoU",
    log_interval=10,
    cv_num=0,
):

    # n_train = len(src_train_data) + len(trg_train_data)

    src_train_loader = DataLoader(
        src_train_data,
        sampler=ImbalancedDatasetSampler(src_train_data),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
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
        best_model_info = {"step": 0, "val": float("inf")}
    elif mode == "max":
        best_model_info = {"step": 0, "val": float("-inf")}

    for step in range(1, steps + 1):
        # ===== Train ===== #
        net.train()

        if (step - 1) % len(src_train_loader) == 0:
            src_data_iter = iter(src_train_loader)
        if (step - 1) % len(trg_train_loader) == 0:
            trg_data_iter = iter(trg_train_loader)
        src_batch = next(src_data_iter)
        trg_batch = next(trg_data_iter)

        src_imgs, trg_imgs = src_batch['image'], trg_batch['image']
        src_labels, trg_labels = src_batch['label'], trg_batch['label']

        imgs = torch.cat((src_imgs, trg_imgs), 0)
        labels = torch.cat((src_labels, trg_labels), 0)
        imgs = imgs.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.long)

        preds = net(imgs)

        loss = criterion(preds, labels)
        step_loss = loss.item() / labels.shape[0]  # ?????????: labels.shape
        logging.info("\n Loss   (train, step):  {}".format(step_loss))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(net.parameters(), 0.1)
        optimizer.step()

        # ===== Validation ===== #
        if (step % log_interval == 0) or (step == 1):  # calculate validation loss and confusion matrix
            val_loss, cm = eval_net_iter(net, val_loader, criterion, device)
            # calculate validation metircs
            val_metrics = eval_metrics(cm)

            if stop_cond == "val_loss":
                cond_val = val_loss
            else:
                cond_val = val_metrics[stop_cond]

            best_model_info = update_best_model(cond_val, step, best_model_info, mode=mode)
            logging.info("\n Loss   (valid, batch): {}".format(val_loss))
            logging.info("\n Acc    (valid, step):  {}".format(val_metrics['accuracy']))
            logging.info("\n Prec   (valid, step):  {}".format(val_metrics['precision']))
            logging.info("\n Recall (valid, step):  {}".format(val_metrics['recall']))
            logging.info("\n mIoU   (valid, step):  {}".format(val_metrics['mIoU']))

            if writer is not None:
                # upload loss (train) and learning_rate to tensorboard
                writer.add_scalar("Loss/train", step_loss, step)

                # # upload confusion_matrix (validation) to tensorboard
                # cm_plt = plot_confusion_matrix(cm, classes, normalize=True)
                # cm_nd = convert_plt2nd(cm_plt)
                # writer.add_image(
                #     "confusion_matrix/valid", cm_nd, global_step=step, dataformats="HWC"
                # )
                # plt.clf()
                # plt.close()

                # # upload not-normed confusion_matrix (validation) to tensorboard
                # cm_plt = plot_confusion_matrix(cm, classes, normalize=False)
                # cm_nd = convert_plt2nd(cm_plt)
                # writer.add_image(
                #     "confusion_matrix_nn/valid", cm_nd, global_step=step, dataformats="HWC"
                # )
                # plt.clf()
                # plt.close()

                # upload loss & score (validation) to tensorboard
                writer.add_scalar("Loss/valid", val_loss, step)
                writer.add_scalar("mIoU/valid", val_metrics['mIoU'], step)
                writer.add_scalar("Accuracy/valid", val_metrics['accuracy'], step)
                writer.add_scalar("Precision/valid", val_metrics['precision'], step)
                writer.add_scalar("Recall/valid", val_metrics['recall'], step)
                writer.add_scalar("F1/valid", val_metrics['f1'], step)

            if best_model_info['step'] == step:
                torch.save(
                    net.state_dict(),
                    checkpoint_dir + f"cv{cv_num}_step{step + 1}.pth",
                )
                logging.info(f"Checkpoint {step + 1} saved !")

            if early_stop(cond_val, step, best_model_info, patience=patience, mode=mode):
                break

    if writer is not None:
        writer.close()


def update_best_model(val, step, best_model_info, mode="max"):
    if mode == "min":
        if val < best_model_info['val']:
            best_model_info['val'] = val
            best_model_info['step'] = step
            print(
                f"[Best Model] step: {best_model_info['step']}, \
                val: {best_model_info['val']}"
            )
    elif mode == "max":
        if val > best_model_info['val']:
            best_model_info['val'] = val
            best_model_info['step'] = step
            print(
                f"[Best Model] step: {best_model_info['step']}, \
                val: {best_model_info['val']}"
            )
    else:
        sys.exit("select mode max or min")
    return best_model_info


def early_stop(val, step, best_model_info, patience=5, mode="max"):
    terminate = False
    if (step - best_model_info['step']) == patience:
        if mode == "min":
            if val >= best_model_info['val']:
                terminate = True
        elif mode == "max":
            if val <= best_model_info['val']:
                terminate = True
        else:
            sys.exit("select mode max or min")
    return terminate


# source + 1??????target?????????
def main_finetune():
    fix_seed(0)
    config_path = "../ST/config_st1_cl[0, 1, 2]_valt3_iter.yaml"

    with open(config_path) as file:
        config = yaml.safe_load(file.read())

    input_shape = tuple(config["main"]["shape"])
    transform = {"Resize": True, "HFlip": True, "VFlip": True}

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    # WSI????????????????????? (target)
    trg_train_wsis = joblib.load(
        config['dataset']['trg_jb_dir']
        + f"{config['main']['trg_facility']}/"
        + "trg_l_wsi.jb"
    )
    trg_valid_wsis = joblib.load(
        config['dataset']['trg_jb_dir']
        + f"{config['main']['trg_facility']}/"
        + "valid_wsi.jb"
    )
    trg_test_wsis = joblib.load(
        config['dataset']['trg_jb_dir']
        + f"{config['main']['trg_facility']}/"
        + "trg_unl_wsi.jb"
    )

    for cv_num in range(config["main"]["cv"]):
        for trg_selected_wsi in trg_train_wsis:
            logging.info(f"== CV{cv_num}: {trg_selected_wsi} ==")
            writer = SummaryWriter(
                log_dir=(
                    (
                        f"{config['main']['result_dir']}logs/st1_val-t3_{config['main']['src_facility']}_"
                        + f"{trg_selected_wsi}_{config['main']['model']}_batch{config['main']['batch_size']}_"
                        + f"shape{config['main']['shape']}_cl{config['main']['classes']}_cv{cv_num}"
                    )
                )
            )

            # ??????????????????
            net = build_model(
                config["main"]["model"], num_classes=len(config["main"]["classes"])
            )
            net.to(device=device)
            # ??????????????????????????????????????????
            if config["main"]["load_pretrained_weight"]:
                weight_path = (
                    config["main"]["pretrained_weight_dir"]
                    + config["main"]["pretrained_weight_names"][cv_num]
                )
                net.load_state_dict(torch.load(weight_path, map_location=device))
                print(f"load_weight: {weight_path}")

            # WSI????????????????????? (source)
            src_train_wsis = joblib.load(
                config["main"]["src_jb_dir"]
                + f"{config['main']['src_facility']}/"
                + f"cv{cv_num}_"
                + f"train_{config['main']['src_facility']}_wsi.jb"
            )

            dataset = WSIDatasetST1_ValT(
                trg_train_wsi=trg_selected_wsi,
                src_train_wsis=src_train_wsis,
                trg_valid_wsis=trg_valid_wsis,
                trg_test_wsis=trg_test_wsis,
                src_imgs_dir=config["dataset"]["src_imgs_dir"],
                trg_imgs_dir=config["dataset"]["trg_imgs_dir"],
                classes=config["main"]["classes"],
                shape=input_shape,
                transform=transform,
            )

            src_train_data, trg_train_data, valid_data, test_data = dataset.get()
            train_wsi, valid_wsi, test_wsi = dataset.get_wsi_split()

            logging.info(
                f"""Starting training:
                Classes:           {config['main']['classes']}
                Epochs:            {config['main']['epochs']}
                Batch size:        {config['main']['batch_size']}
                Model:             {config['main']['model']}
                Optim:             {config['main']['optim']}
                Transform:         {json.dumps(transform)}
                Training size(src):{len(src_train_data)}
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
                + f"st1_val-t3_{config['main']['src_facility']}_{trg_selected_wsi}_{config['main']['classes']}/")
            try:
                os.mkdir(checkpoint_dir)
                logging.info("Created checkpoint directory")
            except OSError:
                pass

            try:
                train_net(
                    net=net,
                    src_train_data=src_train_data,
                    trg_train_data=trg_train_data,
                    valid_data=valid_data,
                    epochs=config["main"]["epochs"],
                    batch_size=config["main"]["batch_size"],
                    device=device,
                    classes=config["main"]["classes"],
                    checkpoint_dir=checkpoint_dir,
                    writer=writer,
                    patience=config["main"]["patience"],
                    stop_cond=config["main"]["stop_cond"],
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
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    main_finetune()
