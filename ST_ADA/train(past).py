import os
from pathlib import Path
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import yaml
import logging
import joblib

from model import Discriminator, Encoder
from dataset import Source_Dataset, Target_Dataset
from eval import eval_net_train, eval_net_test, tensorboard_logging
from util import fix_seed


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train():
    fix_seed(0)
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    # ==== load config ===== #
    # config_path = "./config/config.yaml"
    config_path = "../config/config.yaml"
    with open(config_path) as file:
        config = yaml.safe_load(file.read())

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    classes = config['main']['classes']
    epochs = config['main']['epochs']
    batch = config['main']['batch']
    encoder_name = config['main']['model']
    learning_rate = config['main']['learning_rate']
    shape = tuple(config['main']['shape'])
    cv_num = config['main']['cv_num']
    num_classes = len(classes)

    dataset_dir = config['dataset']['dataset_dir']
    jb_dir = config['dataset']['jb_dir']
    source = config['dataset']['source']
    target = config['dataset']['target']

    weight_path_E = config['weight']['weight_path_E']

    result_dir = config['result']['result_dir']

    transform = {'Resize': True, 'HFlip': True, 'VFlip': True}
    test_transform = {'Resize': True, 'HFlip': False, 'VFlip': False}
    criterion = nn.CrossEntropyLoss()
    num_workers = 2

    # ===== set save weight dir ==== #
    save_D_weight_path = Path(f"{result_dir}/checkpoints/Discriminator/best.pth")
    save_D_weight_path.parent.mkdir(parents=True, exist_ok=True)
    save_D_weight_path.parent.joinpath(f"epoch_weight_src-{source}_trg-{target}").mkdir(parents=True, exist_ok=True)
    save_E_weight_path = Path(f"{result_dir}/checkpoints/Encoder/best.pth")
    save_E_weight_path.parent.mkdir(parents=True, exist_ok=True)
    save_E_weight_path.parent.joinpath(f"epoch_weight_src-{source}_trg-{target}").mkdir(parents=True, exist_ok=True)
    # ============================== #

    project = f"ADA_src-{source}_trg-{target}_{encoder_name}_cl{classes}_cv{cv_num}"

    logging.info(project)

    writer = SummaryWriter(
        log_dir=f"{result_dir}logs/{project}"
    )

    netE = Encoder(
        encoder_name, num_classes, pretrained=False, weight_path=weight_path_E, device=device).to(device)
    # if os.path.exists(weight_path_E):
    #     logging.info(f"load weight_path_E: {weight_path_E}")
    #     netE.load_state_dict(torch.load(weight_path_E, map_location=device))

    netD = Discriminator(encoder_name, num_classes).to(device)
    optE = torch.optim.Adam(netE.parameters(), lr=learning_rate)
    optD = torch.optim.Adam(netD.parameters(), lr=learning_rate)

    # source, targetのWSIリストを読み込み
    train_source_wsi = joblib.load(
        jb_dir
        + f"source_{source}/"
        + f"cv{cv_num}_train_source-{source}_wsi.jb"
    )
    train_target_wsi = joblib.load(
        jb_dir
        + f"target_{target}/"
        + f"cv{cv_num}_train_target-{target}_wsi.jb"
    )
    # validationデータもtraining setsに加える
    train_target_wsi += joblib.load(
        jb_dir
        + f"target_{target}/"
        + f"cv{cv_num}_valid_target-{target}_wsi.jb"
    )

    test_source_wsi = joblib.load(
        jb_dir
        + f"source_{source}/"
        + f"cv{cv_num}_test_source-{source}_wsi.jb"
    )
    test_target_wsi = joblib.load(
        jb_dir
        + f"target_{target}/"
        + f"cv{cv_num}_test_target-{target}_wsi.jb"
    )

    # ===== For Train ===== #
    train_source_dataset = Source_Dataset(
        train_source_wsi,
        imgs_dir=f"{dataset_dir}{source}/",
        classes=classes,
        shape=shape,
        transform=transform,
        mode="train"
    )
    train_target_dataset = Target_Dataset(
        train_target_wsi,
        imgs_dir=f"{dataset_dir}{target}/",
        classes=classes,
        shape=shape,
        transform=transform,
        mode="train"
    )

    train_source_loader = torch.utils.data.DataLoader(
        train_source_dataset, batch_size=batch, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    train_target_loader = torch.utils.data.DataLoader(
        train_target_dataset, batch_size=batch, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    num_train_source = train_source_dataset.__len__()
    num_train_target = train_target_dataset.__len__()
    logging.info(f"num_train_source: {num_train_source}")
    logging.info(f"num_train_target: {num_train_target}")

    # ===== For Test ===== #
    test_source_dataset = Source_Dataset(
        test_source_wsi,
        imgs_dir=f"{dataset_dir}{source}/",
        classes=classes,
        shape=shape,
        transform=test_transform,
        mode="test"
    )
    test_target_dataset = Target_Dataset(
        test_target_wsi,
        imgs_dir=f"{dataset_dir}{target}/",
        classes=classes,
        shape=shape,
        transform=test_transform,
        mode="test"
    )

    test_source_loader = torch.utils.data.DataLoader(
        test_source_dataset, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_target_loader = torch.utils.data.DataLoader(
        test_target_dataset, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    num_test_source = test_source_dataset.__len__()
    num_test_target = test_target_dataset.__len__()
    logging.info(f"num_test_source: {num_test_source}")
    logging.info(f"num_test_target: {num_test_target}")

    source_label = 1
    target_label = 0

    losses_E = []
    losses_A = []
    losses_D = []
    # num of iteration in each epoch (less one would be num_iter)
    num_iter = num_train_source if num_train_source < num_train_target else num_train_target
    logging.info(f"num_iter: {num_iter}")
    for epoch in range(epochs):
        netE.train()
        netD.train()
        pbar = tqdm(total=num_iter)
        loss_E_value = 0
        loss_A_target_value = 0
        loss_D_value = 0

        source_train_cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
        target_train_cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
        d_train_cm = np.zeros((2, 2), dtype=np.int64)  # for Discriminator
        optE.zero_grad()
        optD.zero_grad()
        for i, (source_data, target_data) in enumerate(zip(train_source_loader, train_target_loader)):
            source_img = source_data["img"].to(device)
            source_gt = source_data["gt"].to(device)
            label_source = torch.full((source_img.data.size()[0],), source_label,
                                      dtype=torch.long, device=device)

            target_img = target_data["img"].to(device)
            target_gt = target_data["gt"].to(device)
            label_source_for_target = torch.full((target_img.data.size()[0],), source_label,
                                                 dtype=torch.long, device=device)
            label_target = torch.full((target_img.data.size()[0],), target_label,
                                      dtype=torch.long, device=device)

            # ===== train Encoder ===== #
            # don't accumulate grads in D
            for param in netD.parameters():
                param.requires_grad = False

            # train with source
            source_pred = netE(source_img, mode="class")
            source_loss = criterion(source_pred, source_gt)
            loss = 1000 * source_loss / batch  # 係数(1000)は要調整
            loss.backward()
            loss_E_value += source_loss / batch

            # train with target
            target_featuremap = netE(target_img, mode="feature")
            D_out_target = netD(target_featuremap)
            target_loss = criterion(D_out_target, label_source_for_target)
            loss = target_loss / batch
            loss.backward()
            loss_A_target_value += target_loss / batch

            target_pred = netE(target_img, mode="class")  # for eval train

            #  ====== train Discriminator ====== #
            # bring back requires_grad
            for param in netD.parameters():
                param.requires_grad = True

            # train with source
            source_featuremap = netE(source_img, mode="feature")
            source_featuremap = source_featuremap.detach()  # stop updating Encoder(CNN)
            D_out_source = netD(source_featuremap)
            loss_D_source = criterion(D_out_source, label_source)
            loss_D_source = loss_D_source / batch
            loss_D_source.backward()
            loss_D_value += loss_D_source.item()

            # train with target
            target_featuremap = target_featuremap.detach()
            D_out_target = netD(target_featuremap)
            loss_D_target = criterion(D_out_target, label_target)
            loss_D_target = loss_D_target / batch
            loss_D_target.backward()
            loss_D_value += loss_D_target.item()

            source_train_cm, target_train_cm, d_train_cm = \
                eval_net_train(
                    source_pred=source_pred,
                    source_gt=source_gt,
                    source_d_pred=D_out_source,
                    source_d_label=label_source,
                    target_pred=target_pred,
                    target_gt=target_gt,
                    target_d_pred=D_out_target,
                    target_d_label=label_target,
                    source_cm=source_train_cm,
                    target_cm=target_train_cm,
                    d_cm=d_train_cm
                )

            pbar.update(batch)

        optE.step()
        optD.step()
        losses_E.append(loss_E_value)
        losses_A.append(loss_A_target_value)
        losses_D.append(loss_D_value)

        writer = tensorboard_logging(
            writer,
            epoch,
            loss_E_value,
            loss_A_target_value,
            loss_D_value,
            source_train_cm,
            target_train_cm,
            d_train_cm,
            classes=classes,
            mode="train"
        )

        # ===== Evaluate test data ===== #
        # loss_E_value_t, loss_D_source_value_t, source_cm = \
        #     eval_source_net(
        #         netE,
        #         netD,
        #         test_source_loader,
        #         criterion,
        #         device,
        #         batch,
        #         source_label=1,
        #     )
        # loss_A_target_value_t, loss_D_target_value_t, target_cm = \
        #     eval_target_net(
        #         netE,
        #         netD,
        #         test_target_loader,
        #         criterion,
        #         device,
        #         batch,
        #         source_label=1,
        #         target_label=0
        #     )
        # loss_D_value_t = loss_D_source_value_t + loss_D_target_value_t

        loss_E_value_t, loss_A_target_value_t, loss_D_value_t, source_test_cm, target_test_cm, d_test_cm = \
            eval_net_test(
                netE,
                netD,
                test_source_loader,
                test_target_loader,
                criterion,
                device,
                batch,
                classes,
                source_label=1,
                target_label=0
            )

        writer = tensorboard_logging(
            writer,
            epoch,
            loss_E_value_t,
            loss_A_target_value_t,
            loss_D_value_t,
            source_test_cm,
            target_test_cm,
            d_test_cm,
            classes=classes,
            mode="test"
        )

        pbar.close()

        logging.info(f"iter: {epoch}/{epochs}, loss_E: {loss_E_value_t}, loss_A: {loss_A_target_value_t}, loss_D: {loss_D_value_t}")
        # torch.save(
        #     netE.state_dict(),
        #     str(
        #         save_E_weight_path.parent.joinpath(
        #             f"epoch_weight_src-{source}_trg-{target}/cv{cv_num}_E_epoch{epoch:03d}.pth"
        #         )
        #     ),
        # )
        # torch.save(
        #     netD.state_dict(),
        #     str(
        #         save_D_weight_path.parent.joinpath(
        #             f"epoch_weight_src-{source}_trg-{target}/cv{cv_num}_E_epoch{epoch:03d}.pth"
        #         )
        #     ),
        # )
        if epoch % 5 == 0:
            torch.save(
                netE.state_dict(),
                str(
                    save_E_weight_path.parent.joinpath(
                        f"epoch_weight_src-{source}_trg-{target}/cv{cv_num}_E_epoch{epoch:03d}.pth"
                    )
                ),
            )
            torch.save(
                netD.state_dict(),
                str(
                    save_D_weight_path.parent.joinpath(
                        f"epoch_weight_src-{source}_trg-{target}/cv{cv_num}_D_epoch{epoch:03d}.pth"
                    )
                ),
            )


if __name__ == "__main__":
    train()
