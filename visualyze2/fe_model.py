import os
import sys
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from S.model import build_model
from ST_ADA.model import Encoder


# resnet50から特徴量抽出 (既存)
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


class netE_midlayer(nn.Module):
    def __init__(self, encoder_name: str, num_classes: int = 3, weight_path: str = None, device=torch.device('cuda')):
        super(netE_midlayer, self).__init__()
        netE = Encoder(encoder_name=encoder_name, num_classes=num_classes)

        if weight_path is not None:
            netE.load_state_dict(torch.load(weight_path, map_location=device))
        self.cnn = netE.cnn
        self.pool = netE.pool
        self.in_features = netE.fc.in_features

    def forward(self, x):
        x = self.cnn(x)
        x = self.pool(x)
        x = x.view(-1, self.in_features)
        return x


if __name__ == "__main__":
    num_classes = 3
