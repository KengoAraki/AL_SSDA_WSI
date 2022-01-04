import sys
from torchvision import models
from torch import nn
import torch


in_features = {
    "mobilenet_v2": 1280,
    "inception_v3": 2048,
    "resnet18": 512,
    "resnet50": 2048,
    "resnet101": 2048,
    "wide_resnet101": 2048,
    "alexnet": 4096
}


def build_model(model_name, num_classes, pretrained=True):
    if model_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(in_features=in_features["resnet18"], out_features=num_classes, bias=True)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(in_features=in_features["resnet50"], out_features=num_classes, bias=True)
    elif model_name == "resnet101":
        model = models.resnet101(pretrained=pretrained)
        model.fc = nn.Linear(in_features=in_features["resnet101"], out_features=num_classes, bias=True)
    elif model_name == "wide_resnet101":
        model = models.wide_resnet101_2(pretrained=pretrained)
        model.fc = nn.Linear(in_features=in_features["wide_resnet101"], out_features=num_classes, bias=True)
    else:
        raise Exception("Unexpected model_name: {}".format(model_name))
    return model


class Encoder(nn.Module):
    def __init__(self, encoder_name, num_classes, pretrained=False, weight_path=None, device='cuda'):
        super().__init__()
        self.encoder_name = encoder_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.weight_path = weight_path
        self.device = device
        self.cnn, self.fc, self.pool = \
            self.select_base_model(self.encoder_name)

    def forward(self, x, mode="feature"):
        x = self.cnn(x)
        x = self.pool(x)
        x = x.view(-1, self.fc.in_features)
        if mode == "class":
            x = self.fc(x)
        elif mode == "feature":
            pass
        else:
            sys.exit("select correct mode")
        return x

    def load_model(self, encoder_name):
        # デフォルトの事前学習済みの重み(ImageNet)を読み込む場合
        if self.pretrained is True:
            base_model = build_model(encoder_name, self.num_classes, pretrained=True)
            if self.weight_path is not None:
                base_model.load_state_dict(torch.load(self.weight_path, map_location=self.device))
        else:
            base_model = build_model(encoder_name, self.num_classes, pretrained=False)
        return base_model

    def select_base_model(self, encoder_name):
        model = self.load_model(encoder_name)
        if encoder_name == "resnet18":
            model.fc = nn.Linear(in_features=in_features["resnet18"], out_features=self.num_classes, bias=True)
            cnn = nn.Sequential(*list(model.children()))[:-2]
            fc = model.fc
            pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        elif encoder_name == "resnet50":
            model.fc = nn.Linear(in_features=in_features["resnet50"], out_features=self.num_classes, bias=True)
            cnn = nn.Sequential(*list(model.children()))[:-2]
            fc = model.fc
            pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        elif encoder_name == "resnet101":
            model.fc = nn.Linear(in_features=in_features["resnet101"], out_features=self.num_classes, bias=True)
            cnn = nn.Sequential(*list(model.children()))[:-2]
            fc = model.fc
            pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        elif encoder_name == "wide_resnet101":
            model.fc = nn.Linear(in_features=in_features["wide_resnet101"], out_features=self.num_classes, bias=True)
            cnn = nn.Sequential(*list(model.children()))[:-2]
            fc = model.fc
            pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        else:
            raise Exception("Unexpected encoder_name: {}".format(encoder_name))
        return cnn, fc, pool


class Discriminator(nn.Module):
    def __init__(self, encoder_name: str, num_classes: int = 2):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features=in_features[encoder_name], out_features=num_classes, bias=True),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class Discriminator2(nn.Module):
    def __init__(self, encoder_name: str, num_classes: int = 2):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features=in_features[encoder_name], out_features=500, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=500, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=num_classes, bias=True),
        )

    def forward(self, x):
        x = self.layers(x)
        return x
