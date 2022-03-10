import os
import sys
import yaml
import joblib
from tqdm import tqdm
import collections

from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from S.dataset import WSIDataset
from S.util import fix_seed, ImbalancedDatasetSampler2



if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    fix_seed(0)
    config_path = "./S/config_s_MF0003_cl[0, 1, 2].yaml"

    with open(config_path) as file:
        config = yaml.safe_load(file.read())
    cv_num = 0
    input_shape = tuple(config['main']['shape'])
    transform = {"Resize": True, "HFlip": True, "VFlip": True}
    batch_size = 32
    epochs = 10

    # WSIのリストを取得 (source)
    train_wsis = joblib.load(
        config['dataset']['jb_dir']
        + f"{config['main']['facility']}/"
        + f"cv{cv_num}_"
        + f"train_{config['main']['facility']}_wsi.jb"
    )
    valid_wsis = joblib.load(
        config['dataset']['jb_dir']
        + f"{config['main']['facility']}/"
        + f"cv{cv_num}_"
        + f"valid_{config['main']['facility']}_wsi.jb"
    )
    test_wsis = joblib.load(
        config['dataset']['jb_dir']
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

    n_train = len(train_data)

    train_loader = DataLoader(
        train_data,
        sampler=ImbalancedDatasetSampler2(train_data),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    for epoch in range(epochs):
        with tqdm(
            total=n_train, desc=f"Epoch {epoch + 1}/{epochs}", unit="img"
        ) as pbar:
            # 短いdataloaderに合わせる
            count_label = []  # to check imblanaced sampler
            for batch in train_loader:
                imgs = batch['image']
                labels = batch['label']
                count_label.extend(labels.tolist())  # to check sampler
                pbar.update(imgs.shape[0])
        print(f"sample balance: {collections.Counter(count_label)}")  # to check sampler