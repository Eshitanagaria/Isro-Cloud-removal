import torch
import torch.nn.functional as F


# S2 order: [R, G, B, NIR] -> [0, 1, 2, 3]
BAND_IDX = {'red': 0, 'green': 1, 'nir': 3}


def ndvi(x, eps=1e-6):
    red = x[:, BAND_IDX['red']]
    nir = x[:, BAND_IDX['nir']]
    return (nir - red) / (nir + red + eps)


def ndwi(x, eps=1e-6):
    green = x[:, BAND_IDX['green']]
    nir = x[:, BAND_IDX['nir']]
    return (green - nir) / (green + nir + eps)


def spectral_consistency_loss(pred, target):
    l_ndvi = F.l1_loss(ndvi(pred), ndvi(target))
    l_ndwi = F.l1_loss(ndwi(pred), ndwi(target))
    return l_ndvi + l_ndwi

if __name__ == "__main__":
    pred = torch.rand(2, 4, 256, 256)
    target = torch.rand(2, 4, 256, 256)
    loss = spectral_consistency_loss(pred, target)
    print("Loss value:", loss.item())
