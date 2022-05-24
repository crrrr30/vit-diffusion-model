import numpy as np
import glob
import torch
from tqdm import trange
from model import UNet

model = UNet()
checkpoint = torch.load(
    sorted(glob.glob('../Downloads/checkpoint*.pkl')).pop(),
    map_location = lambda _, __: _
)
model.load_state_dict(checkpoint['model_state_dict'])

xprev = None
x = torch.randn(1, 3, 256, 256)
beta = np.linspace(0.001, 0.2, 1000)
alpha = 1 - beta
bar = trange(999,-1,-1)

model.eval()

with torch.no_grad():
    for t in bar:
        xprev = x.clone()
        x = model(x)
        bar.set_description(f'{x.max().item():.04f},{x.min().item():.04f}')
        if x.isnan().any().item(): break

    import matplotlib.pyplot as plt
    plt.imshow(x[0].transpose(2, 0).numpy())
    plt.show()

