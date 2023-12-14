from diffusers import UNet2DModel, DDIMScheduler, VQModel
import torch
import PIL.Image
import numpy as np
import tqdm
from PIL import Image
from torchvision import transforms
import torchvision.utils as tvu
import os

def MeanUpsample(x, scale):
    n, c, h, w = x.shape
    out = torch.zeros(n, c, h, scale, w, scale).to(x.device) + x.view(n,c,h,1,w,1)
    out = out.view(n, c, scale*h, scale*w)
    return out

def load_img_cvt_tensor(img_path):
    """
    input : img path
    output : img tensor with shape (1, c, h, w)
    """
    img = Image.open(img_path).convert("RGB")
    img = np.asarray(img)/255

    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((256,256))])
    # transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img).type(torch.float)
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor

def data_transform(X):
    X = 2 * X - 1.0
    return X


def inverse_data_transform(X):
    X = (X + 1.0) / 2.0
    return torch.clamp(X, 0.0, 1.0)

scale = 4
A = torch.nn.AdaptiveAvgPool2d((256//scale,256//scale))
Ap = lambda z: MeanUpsample(z,scale)

x_orig = load_img_cvt_tensor('/eva_data2/shlu2240/DDNM/exp/datasets/old_celeba_hq/face/00234.png').cuda()
x_orig = 2 * x_orig - 1.0
y = A(x_orig)
Apy = Ap(y)
os.makedirs(os.path.join("./exp/image_samples/ldm", "Apy"), exist_ok=True)
for i in range(len(Apy)):
    tvu.save_image(
        inverse_data_transform(Apy[i]),
        os.path.join("./exp/image_samples/ldm", f"Apy/Apy.png")
    )
    tvu.save_image(
        inverse_data_transform(x_orig[i]),
        os.path.join("./exp/image_samples/ldm", f"Apy/orig.png")
    )



seed = 3

# 1. Unroll the full loop
# ==================================================================
# load all models
unet = UNet2DModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="unet")
vqvae = VQModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="vqvae")
scheduler = DDIMScheduler.from_config("CompVis/ldm-celebahq-256", subfolder="scheduler")

# set to cuda
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

unet.to(torch_device)
vqvae.to(torch_device)


# generate gaussian noise to be decoded
generator = torch.manual_seed(seed)
x = torch.randn(
    (1, unet.config.in_channels, 64, 64),
    generator=generator,
).to(torch_device)

# set inference steps for DDIM
scheduler.set_timesteps(num_inference_steps=200)

# Code form RePaint   
def get_schedule_jump(T_sampling, travel_length, travel_repeat):
    jumps = {}
    for j in range(0, T_sampling - travel_length, travel_length):
        jumps[j] = travel_repeat - 1

    t = T_sampling
    ts = []

    while t >= 1:
        t = t-1
        ts.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1
                ts.append(t)

    ts.append(-1)

    _check_times(ts, -1, T_sampling)
    return ts
def _check_times(times, t_0, T_sampling):
    # Check end
    assert times[0] > times[1], (times[0], times[1])

    # Check beginning
    assert times[-1] == -1, times[-1]

    # Steplength = 1
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)

    # Value range
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= T_sampling, (t, T_sampling)

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


image = x
for t in tqdm.tqdm(scheduler.timesteps):
    # predict noise residual of previous image
    with torch.no_grad():
        residual = unet(image, t)["sample"]


    # compute previous image x_t according to DDIM formula
    prev_image = scheduler.step(residual, t, image, eta=0.0)["prev_sample"]

    # x_t-1 -> x_t
    image = prev_image

# decode image with vae
with torch.no_grad():
    image = vqvae.decode(image)

# process image
image_processed = image.sample.cpu().permute(0, 2, 3, 1)
image_processed = (image_processed + 1.0) * 127.5
image_processed = image_processed.clamp(0, 255).numpy().astype(np.uint8)
image_pil = PIL.Image.fromarray(image_processed[0])
image_pil.save('output.png')