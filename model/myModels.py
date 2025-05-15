import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
        - VAE Encoder
    """
    def __init__(self):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(512 * 2, 2048),
            nn.ReLU())
        self.mean = nn.Linear(2048, 512)
        self.log_var = nn.Linear(2048, 512)
        self.apply(weights_init)
        
    def forward(self, x, c=None):
        if c is not None:
            x = torch.cat([x, c], dim=1)
        x = self.net(x)
        mean = self.mean(x)
        log_var = self.log_var(x)
        return mean, log_var


class Decoder(nn.Module):
    """
        - VAE Decoder
    """
    def __init__(self):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(512*2, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512))
        self.apply(weights_init)
    
    def forward(self, x):
        out = self.net(x)
        return out


class Myclassifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        """
            - in_dim:input dimension
            - out_dim: class categories
        """
        super().__init__()
        self.fc1 = nn.Linear(in_dim, int(in_dim / 2))
        self.fc2 = nn.Linear(int(in_dim / 2), int(in_dim / 2))
        self.fc3 = nn.Linear(int(in_dim / 2), out_dim)
        self.drop = nn.Dropout()
        self.leakyRelu =  nn.LeakyReLU()
        self.apply(weights_init)

    def forward(self, x):
        for fc in [self.fc1, self.fc2]:
            x = self.leakyRelu(fc(x))
            x = self.drop(x)
        x = self.fc3(x)
        return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)   