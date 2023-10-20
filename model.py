import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, backbone):
        super(Encoder, self).__init__()

        # self.backbone = backbone
        self.backbone = nn.Sequential(
            backbone, 
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten()
        )
    
    def forward(self, x):
        x = self.backbone(x)

        return x


class Classifier(nn.Module):
    def __init__(self, backbone, hdim=512, n_class=10, reg=True):
        super(Classifier, self).__init__()

        # self.backbone = backbone
        self.backbone = nn.Sequential(
            backbone, 
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten()
        )
        self.predict = nn.Sequential(
            nn.Linear(backbone.out_features, hdim),
            nn.BatchNorm1d(hdim),
            nn.ReLU(),
            nn.Linear(hdim, n_class)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.predict(x)

        return x
    
    def get_parameters(self, base_lr=1.0):
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr},
            {"params": self.predict.parameters(), "lr": 1.0 * base_lr}
        ]

        return params


class ENCODER(nn.Module):
    def __init__(self, rgb=False, resnet=False):
        super(ENCODER, self).__init__()

        if rgb:
            self.encode = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding="same"),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding="same"),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding="same"),
                nn.ReLU(),
            )
        else:
            self.encode = nn.Sequential(
                    nn.Conv2d(1, 32, 3, padding="same"),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, 3, padding="same"),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, 3, padding="same"),
                    nn.ReLU(),
                )

        
    def forward(self, x):
        x = self.encode(x)
        return x
    
    
class MLP(nn.Module):
    def __init__(self, mode, n_class, hidden=1024):
        super(MLP, self).__init__()

        if mode == "mnist":
            dim = 25088
        elif mode == "portraits":
            dim = 32768
        else:
            dim = 2048

        if mode == "covtype":
            hidden = 256
            self.mlp = nn.Sequential(
                nn.Linear(54, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.BatchNorm1d(hidden),
                nn.Linear(hidden, n_class)
            )
        else:
            hidden = 128
            self.mlp = nn.Sequential(
                # nn.BatchNorm2d(32),
                nn.Flatten(),
                # nn.Linear(dim, n_class),
                nn.Linear(dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.BatchNorm1d(hidden),
                nn.Linear(hidden, n_class)
            )
        
    def forward(self, x):
        return self.mlp(x)


class Classifier(nn.Module):
    def __init__(self, encoder, mlp):
        super(Classifier, self).__init__()

        self.encoder = encoder
        self.mlp = mlp
        
    def forward(self, x):
        x = self.encoder(x)
        return self.mlp(x)


class MLP_Encoder(nn.Module):
    def __init__(self, hidden=256):
        super(MLP_Encoder, self).__init__()

        self.encode = nn.Sequential(
        )
        
    def forward(self, x):
        return self.encode(x)