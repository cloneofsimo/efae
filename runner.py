import argparse

import torch
import random


class conj_RF:
    def __init__(self, model, encoder, ln=True):
        self.model = model
        self.encoder = encoder
        self.ln = ln

    def forward(self, x, cond):

        # with random chance, flip both x and cond.
        if random.random() < 0.5:
            x = torch.flip(x, [-1])
            cond[:, -4:] = -cond[:, -4:]


        b = x.size(0)
        if self.ln:
            nt = torch.randn((b,)).to(x.device)
            t = torch.sigmoid(nt)
        else:
            t = torch.rand((b,)).to(x.device)
        texp = t.view([b, *([1] * len(x.shape[1:]))])
        z1 = torch.randn_like(x)
        zt = (1 - texp) * x + texp * z1
        vtheta = self.model(zt, t, cond)
        batchwise_mse = ((z1 - x - vtheta) ** 2).mean(dim=list(range(1, len(x.shape[1:]))))
        tlist = batchwise_mse.detach().cpu().reshape(-1).tolist()
        ttloss = [(tv, tloss) for tv, tloss in zip(t, tlist)]
        return batchwise_mse.mean(), ttloss

    @torch.no_grad()
    def sample(self, z, cond, null_cond=None, sample_steps=50, cfg=2.0):
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z]
        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device)

            vc = self.model(z, t, cond)
            if null_cond is not None:
                vu = self.model(z, t, null_cond)
                vc = vu + cfg * (vc - vu)

            z = z - dt * vc
            images.append(z)
        return images

class SomeEncoderModel(torch.nn.Module):
    # good all conv that takes 32x32 image and outputs 256 dim vector
    def __init__(self, channels):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(channels, 32, 4, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 4, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 4, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, 4, 2, 1),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        x = self.conv(x).view(x.size(0), -1)
        print(x.shape)
        return x
    

if __name__ == "__main__":
    # train class conditional conj_RF on mnist.
    import numpy as np
    import torch.optim as optim
    from PIL import Image
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from torchvision.utils import make_grid
    from tqdm import tqdm

    import wandb
    from dit import DiT_Llama

    parser = argparse.ArgumentParser(description="use cifar?")
    parser.add_argument("--cifar", action="store_true")
    args = parser.parse_args()
    CIFAR = args.cifar

    if CIFAR:
        dataset_name = "cifar"
        fdatasets = datasets.CIFAR10
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        channels = 3
        model = DiT_Llama(

            channels, 32, dim=256, n_layers=10, n_heads=8, condition_dim=256
        ).cuda()
        encoder = SomeEncoderModel(channels).cuda()

    else:
        dataset_name = "mnist"
        fdatasets = datasets.MNIST
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Pad(2),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        channels = 1
        model = DiT_Llama(

            channels, 32, dim=64, n_layers=6, n_heads=4, condition_dim=256
        ).cuda()
        encoder = SomeEncoderModel(channels).cuda()

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {model_size}, {model_size / 1e6}M")

    rf = conj_RF(model, encoder)

    all_params = list(model.parameters()) + list(encoder.parameters())
    optimizer = optim.Adam(all_params, lr=2e-3)
    criterion = torch.nn.MSELoss()

    mnist = fdatasets(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(mnist, batch_size=512, shuffle=True, drop_last=True)

    wandb.init(project=f"rf_selfencoder_{dataset_name}")

    for epoch in range(100):
        lossbin = {i: 0 for i in range(10)}
        losscnt = {i: 1e-6 for i in range(10)}
        for i, (x, _) in tqdm(enumerate(dataloader)):
            x = x.cuda()
            cond = rf.encoder(x)  # Encode the input data to get the condition
            # with 10%, make the condition null
            cond = torch.where(
                torch.rand_like(cond[:, :1]) < 0.1,
                torch.zeros_like(cond),
                cond,
            )
            
            optimizer.zero_grad()
            loss, blsct = rf.forward(x, cond)
            loss.backward()
            optimizer.step()

            wandb.log({"loss": loss.item()})

            # count based on t
            for t, l in blsct:
                lossbin[int(t * 10)] += l
                losscnt[int(t * 10)] += 1

        # log
        for i in range(10):
            print(f"Epoch: {epoch}, {i} range loss: {lossbin[i] / losscnt[i]}")

        wandb.log({f"lossbin_{i}": lossbin[i] / losscnt[i] for i in range(10)})

        rf.model.eval()
        with torch.no_grad():
            x = x[:8]
            cond = rf.encoder(x)  # Encode the noise data to get the condition
            cond_flip = cond.clone()
            cond_flip[:, -4:] = -cond_flip[:, -4:]
            cond = torch.cat([cond, cond_flip], dim=0)
            null_cond = torch.zeros_like(cond)  # Assuming null condition is a zero tensor

            init_noise = torch.randn(16, channels, 32, 32).cuda()
            images = rf.sample(init_noise, cond, null_cond)
            # image sequences to gif
            gif = []
            for image in images:
                # unnormalize
                image = image * 0.5 + 0.5
                image = image.clamp(0, 1)
                x_as_image = make_grid(image.float(), nrow=4)
                img = x_as_image.permute(1, 2, 0).cpu().numpy()
                img = (img * 255).astype(np.uint8)
                gif.append(Image.fromarray(img))

            gif[0].save(
                f"contents/sample_{epoch}.gif",
                save_all=True,
                append_images=gif[1:],
                duration=100,
                loop=0,
            )

            last_img = gif[-1]
            last_img.save(f"contents/sample_{epoch}_last.png")

        rf.model.train()
