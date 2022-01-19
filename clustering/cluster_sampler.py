import os
import sys
import torch
from torch import nn
from tqdm import tqdm
from sklearn.cluster import KMeans
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from S.model import build_model
from S.dataset import WSI
from clustering.wsi_cluster_analysis import preprocess_feature_vecs


# resnet50から特徴量抽出
class resnet50_midlayer(nn.Module):
    def __init__(self, num_classes: int = 3, weight_path: str = None, device=torch.device('cuda')):
        super(resnet50_midlayer, self).__init__()
        org_model = build_model("resnet50", num_classes, pretrained=True)

        if weight_path is not None:
            org_model.load_state_dict(torch.load(weight_path, map_location=device))
        self.encoder = nn.Sequential(*(list(org_model.children())[:-1]))

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 2048)
        return x


def extract_feature(model, data_loader, device=torch.device('cuda')):
    model.eval()
    init_flag = True
    n_batch = len(data_loader)
    with torch.no_grad():
        with tqdm(total=n_batch, desc="extract clustering feature", unit="batch") as pbar:
            for batch in data_loader:
                data, target = batch["image"], batch["label"]
                data, target = data.to(device), target.to(device)
                latent_vecs = model(data)

                latent_vecs = latent_vecs.cpu()
                target = target.cpu()
                if init_flag:
                    latent_vecs_stack = latent_vecs
                    target_stack = target
                    init_flag = False
                else:
                    latent_vecs_stack = torch.cat((latent_vecs_stack, latent_vecs), 0)
                    target_stack = torch.cat((target_stack, target), 0)
                pbar.update(1)
    return latent_vecs_stack, target_stack


def get_cluster_ids(
    file_list: list,
    weight_path: str = None,
    cluster_num: int = 10,
    batch_size: int = 16,
    shape: tuple = (256, 256),
    classes: list = [0, 1, 2],
    random_state: int = 0,
) -> list:
    """
    modelはresnet50のみ対応
    """

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 2, "pin_memory": True} if use_cuda else {}

    # 特徴量を抽出モデルを定義 (resnet50のみ対応)
    # weight_pathがNoneの場合，ImageNetの重み
    model = resnet50_midlayer(
        num_classes=len(classes), weight_path=weight_path
    ).to(device=device)

    transform = {"Resize": True, "HFlip": False, "VFlip": False}
    dataset = WSI(
        file_list, classes, shape, transform, is_pred=False
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, **kwargs
    )

    # 特徴量を抽出
    latent_vecs, _ = extract_feature(
        model=model, data_loader=data_loader, device=device
    )
    latent_vecs = latent_vecs.numpy()

    # # 保存
    # np.save(
    #     "latent_vecs",
    #     latent_vecs,
    # )

    # latent_vecsの標準化と次元削減
    feature_vecs = preprocess_feature_vecs(
        latent_vecs,
        reduced_dim=30,
        random_state=random_state,
        standarize=False,
    )

    # KMeansでクラスタリング
    kmeans = KMeans(n_clusters=cluster_num, random_state=random_state)
    clusters = kmeans.fit(feature_vecs)
    cluster_ids = clusters.labels_
    cluster_ids = cluster_ids.tolist()

    c_list = sorted(Counter(cluster_ids).items(), key=lambda x: x[0])
    print(dict(c_list))

    return cluster_ids
