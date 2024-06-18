from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline
from diffusers.image_processor import VaeImageProcessor

device = "cuda:0"
import numpy as np
import torch


def prepare_image(pil_image, w=512, h=512):
    thisw, thish = pil_image.size
    assert thisw == 256 or thish == 256, f"Image size is {thisw}x{thish}"

    arr = np.array(pil_image.convert("RGB"))
    arr = arr.astype(np.float32) / 127.5 - 1
    arr = np.transpose(arr, [2, 0, 1])
    image = torch.from_numpy(arr)
    return image



vae_model = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device)

from PIL import Image

processed_images = prepare_image(Image.open("chubocat.png")).unsqueeze(0).to(device)
vae_outputs = vae_model.encode(processed_images).latent_dist.sample()
print(vae_outputs.shape)
# flip the output
vae_outputs = torch.flip(vae_outputs, dims=[-1])






x = vae_model.decode(vae_outputs.cuda()).sample
img = VaeImageProcessor().postprocess(image=x.detach(), do_denormalize=[True, True])[0]
img.save("test.png")