import os, platform, math, random
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from torchvision import transforms, datasets
from torchvision.utils import save_image
import matplotlib.pyplot as plt

# ------------------------------------------------
# 0.  Global configuration
# ------------------------------------------------
DATA_ROOT          = "data"
IMG_HEIGHT, WIDTH  = 256, 256
DOWNSAMPLE         = 16
BATCH_SIZE         = 16

EPOCHS_VQGAN       = 40
EPOCHS_CLASSIFIER  = 20

# ↓↓↓  NEW ↓↓↓ -----------------------------------
AUGMENT_MULTIPLIER = 0.10   # 10 %   (was 50 %)
REAL_SYNTH_RATIO   = 0.70   # 70 % real, 30 % synthetic **inside a batch**
FID_CUTOFF         = 500.0   # optional; skip bad GANs, pod 50 bi rablo bit ampak...
# -----------------------------------------------

DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Windows needs workers = 0 unless you go through the torch.multiprocessing API
NUM_WORKERS        = 0 if platform.system() == "Windows" else 2
torch.manual_seed(42);  random.seed(42);  np.random.seed(42)

# ------------------------------------------------
# 1.  Dataset helpers
# ------------------------------------------------


# ------------------------------------------------
# FID helper  (put near the top, after imports)
# ------------------------------------------------
def quick_fid(real_dir: str, fake_imgs: torch.Tensor, cls: str) -> float:
    """
    Compute FID between all images in train/<cls> and the Tensor fake_imgs
    (shape [N,3,H,W], values in [0,1]). Returns np.inf if torchmetrics
    isn't installed.
    """
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchvision.datasets import ImageFolder
        from torch.utils.data import DataLoader, Subset
        from torchvision import transforms
    except ImportError:
        return float("inf")

    # 1) initialize the metric
    fid = FrechetInceptionDistance(feature=2048).to(DEVICE)

    # common transform to 299×299 uint8
    tf = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),  # floats in [0,1]
    ])

    # 2) load & update on real images (filtering via ImageFolder)
    #    real_dir should be DATA_ROOT/train
    full_ds   = ImageFolder(real_dir, transform=tf)
    cls_idx   = full_ds.class_to_idx[cls]
    # pick only those samples whose target == cls_idx
    indices  = [i for i, t in enumerate(full_ds.targets) if t == cls_idx]
    real_sub  = Subset(full_ds, indices)
    real_loader = DataLoader(real_sub, batch_size=BATCH_SIZE, num_workers=0)

    for x, _ in real_loader:
        x_uint8 = (x.clamp(0, 1) * 255).to(torch.uint8).to(DEVICE)
        fid.update(x_uint8, real=True)

    # 3) resize & update on fake images
    fake_resized = torch.nn.functional.interpolate(
        fake_imgs, size=(299, 299), mode='bilinear', align_corners=False
    )
    for chunk in fake_resized.split(BATCH_SIZE):
        chunk_uint8 = (chunk.clamp(0, 1) * 255).to(torch.uint8).to(DEVICE)
        fid.update(chunk_uint8, real=False)

    # 4) compute & return FID
    return fid.compute().item()

tf_gan = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
])

tf_cls = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, WIDTH)),
    transforms.ToTensor(),
])

def label_to_tensor(y):          # picklable replacement for the lambda
    return torch.tensor(y)

# ------------------------------------------------
# 2.  VQ-GAN components
# ------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, in_ch=3, latent=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 128, 4, 2, 1), nn.LeakyReLU(.2,True),  # 256→128
            nn.Conv2d(128, 256, 4, 2, 1),  nn.LeakyReLU(.2,True),  # 128→64
            nn.Conv2d(256, 512, 4, 2, 1),  nn.LeakyReLU(.2,True),  # 64→32
            nn.Conv2d(512, latent, 4, 2, 1),nn.LeakyReLU(.2,True), # 32→16
        )
    def forward(self,x): return self.net(x)

class Decoder(nn.Module):
    def __init__(self, out_ch=3, latent=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent,512,4,2,1), nn.LeakyReLU(.2,True),#16→32
            nn.ConvTranspose2d(512,256,4,2,1),  nn.LeakyReLU(.2,True), #32→64
            nn.ConvTranspose2d(256,128,4,2,1),  nn.LeakyReLU(.2,True), #64→128
            nn.ConvTranspose2d(128,out_ch,4,2,1), nn.Tanh(),           #128→256
        )
    def forward(self,z): return self.net(z)

class VectorQuantizer(nn.Module):
    def __init__(self, K=512, latent=256, beta=.25):
        super().__init__(); self.beta=beta
        self.codebook = nn.Embedding(K, latent); nn.init.normal_(self.codebook.weight,0,.02)
    def forward(self,z):
        B,C,H,W = z.shape
        z_flat  = z.permute(0,2,3,1).reshape(-1,C)
        dist    = (z_flat.unsqueeze(1)-self.codebook.weight).pow(2).sum(-1)
        idx     = dist.argmin(1)
        z_q     = self.codebook(idx).view(B,H,W,C).permute(0,3,1,2).contiguous()
        loss    = self.beta*(z_q.detach()-z).pow(2).mean() + (z_q-z.detach()).pow(2).mean()
        z_q     = z + (z_q-z).detach()
        return z_q, idx, loss
    def get_code(self, idx, shape):
        return self.codebook(idx).view(*shape)

class Discriminator(nn.Module):
    def __init__(self, in_c=3, base=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_c, base,4,2,1), nn.LeakyReLU(.2,True),
            nn.Conv2d(base, base*2,4,2,1), nn.LeakyReLU(.2,True),
            nn.Conv2d(base*2,base*4,4,2,1), nn.LeakyReLU(.2,True),
            nn.Conv2d(base*4,base*8,4,2,1), nn.LeakyReLU(.2,True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(base*8,1)
        )
    def forward(self,x): return self.main(x)

class VQGAN(nn.Module):
    def __init__(self, latent=256, K=512):
        super().__init__()
        self.enc = Encoder(latent=latent)
        self.dec = Decoder(latent=latent)
        self.vq  = VectorQuantizer(K, latent)
    def forward(self,x):
        z_e = self.enc(x)
        z_q, idx, vqloss = self.vq(z_e)
        x_rec = self.dec(z_q)
        return x_rec, idx, vqloss

# ------------------------------------------------
# 3.  VQ-GAN training / sampling helpers
# ------------------------------------------------
def train_vqgan(model, disc, loader, epochs, save_dir):
    opt_g = optim.Adam(model.parameters(), lr=1e-4, betas=(.5,.999))
    opt_d = optim.Adam(disc.parameters(),  lr=1e-4, betas=(.5,.999))
    l1 = nn.L1Loss(); bce = nn.BCEWithLogitsLoss()

    model.train(); disc.train()
    for ep in range(epochs):
        g_loss_tot=d_loss_tot=0; real_acc=fake_acc=0
        for real,_ in loader:
            real = real.to(DEVICE)

            # ---- generator ----
            opt_g.zero_grad()
            rec,_,vq_loss = model(real)
            rec_loss      = l1(rec, real)
            d_fake        = disc(rec)
            adv_loss      = bce(d_fake, torch.ones_like(d_fake))
            g_loss        = rec_loss + vq_loss + 0.1*adv_loss
            g_loss.backward(); opt_g.step()

            # ---- discriminator ----
            opt_d.zero_grad()
            d_real = disc(real)
            d_fake_det = disc(rec.detach())
            d_loss = bce(d_real, torch.ones_like(d_real)) + \
                     bce(d_fake_det, torch.zeros_like(d_fake_det))
            d_loss.backward(); opt_d.step()

            # stats
            g_loss_tot += g_loss.item(); d_loss_tot += d_loss.item()
            real_acc += (d_real>0).float().mean().item()
            fake_acc += (d_fake_det<0).float().mean().item()

        print(f"[{save_dir}] Ep {ep+1:03}/{epochs} "
              f"G={g_loss_tot/len(loader):.4f} D={d_loss_tot/len(loader):.4f} "
              f"D(real)={real_acc/len(loader):.2%} D(fake)={fake_acc/len(loader):.2%}")

        with torch.no_grad():
            preview = (rec[:16].clamp(-1,1)+1)/2
            save_image(preview, os.path.join(save_dir,f"recon_ep{ep+1:03}.png"), nrow=4)

@torch.no_grad()
def sample(model, n, latent=256, K=512):
    tokens = (IMG_HEIGHT//DOWNSAMPLE)*(WIDTH//DOWNSAMPLE)
    idx = torch.randint(0, K, (n, tokens), device=DEVICE)
    z_q = model.vq.get_code(idx.view(-1),
                            (n, latent, IMG_HEIGHT//DOWNSAMPLE, WIDTH//DOWNSAMPLE))
    img = model.dec(z_q)
    return ((img+1)/2).cpu()  # [0,1]

# ------------------------------------------------
# 4.  Classifier
# ------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, n_cls):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),   #256→128
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),  #128→64
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2), #64→32
            nn.Conv2d(128,256,3,padding=1), nn.ReLU(), nn.MaxPool2d(2) #32→16
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256,128), nn.ReLU(),
            nn.Linear(128,n_cls)
        )
    def forward(self,x): return self.classifier(self.features(x))

def train_classifier(model, loader_tr, loader_val, epochs):
    crit = nn.CrossEntropyLoss(); opt = optim.Adam(model.parameters(), lr=1e-3)
    patience = 8                 # stop after 8 epochs w/o improvement
    best_val = 0.0
    streak   = 0
    for ep in range(epochs):
        model.train(); running=correct=tot=0
        for x,y in loader_tr:
            x,y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); out = model(x); loss = crit(out,y)
            loss.backward(); opt.step()
            running += loss.item();  pred = out.argmax(1)
            correct += (pred==y).sum().item();  tot += y.numel()

        if ep%10==9 or ep==epochs-1:
            model.eval(); v_c=v_t=0
            with torch.no_grad():
                for x,y in loader_val:
                    x,y=x.to(DEVICE),y.to(DEVICE)
                    v_c+=(model(x).argmax(1)==y).sum().item();  v_t+=y.numel()
                    if v_c/v_t > best_val:
                        best_val = v_c/v_t
                        streak = 0
                    else:
                        streak += 1
                        if streak == patience:
                            print(f"  ↳ early-stopped at epoch {ep+1}")
                            break
            print(f"  CNN Ep{ep+1:03}/{epochs} "
                  f"loss={running/len(loader_tr):.4f} "
                  f"trainAcc={correct/tot:.3%} valAcc={v_c/v_t:.3%}")

def evaluate(model, loader):
    model.eval(); c=t=0
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(DEVICE),y.to(DEVICE)
            c+=(model(x).argmax(1)==y).sum().item(); t+=y.numel()
    return c/t

# ------------------------------------------------
# 5.  Main
# ------------------------------------------------
if __name__ == "__main__":
    # 5-A  discover class names
    train_root   = os.path.join(DATA_ROOT,"train")
    class_names  = sorted([d for d in os.listdir(train_root)
                           if os.path.isdir(os.path.join(train_root,d))])
    print("Classes:", class_names)

    # base datasets for     CNN
    train_base = datasets.ImageFolder(train_root,
                                      transform=tf_cls,
                                      target_transform=label_to_tensor)
    val_base   = datasets.ImageFolder(os.path.join(DATA_ROOT,"valid"),
                                      transform=tf_cls,
                                      target_transform=label_to_tensor)
    test_base  = datasets.ImageFolder(os.path.join(DATA_ROOT,"test"),
                                      transform=tf_cls,
                                      target_transform=label_to_tensor)

    # loaders for plain training
    dl_args = dict(batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    train_loader_base = DataLoader(train_base, **dl_args)
    val_loader        = DataLoader(val_base,  batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    test_loader       = DataLoader(test_base, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # 5-B  train CNN on the **original** data
    cls_plain = SimpleCNN(len(class_names)).to(DEVICE)
    print("\nTraining CNN on original dataset …")
    train_classifier(cls_plain, train_loader_base, val_loader, EPOCHS_CLASSIFIER)
    acc_plain = evaluate(cls_plain, test_loader)
    print(f"Test accuracy (plain): {acc_plain:.3%}\n")

    # 5-C  one GAN per class  + immediate PNG saving
    synth_sets = []
    for cls in class_names:
        print(f"\n-----  VQ-GAN for class: {cls}  -----")

        full_gan_ds = datasets.ImageFolder(train_root, transform=tf_gan)
        cls_idx     = full_gan_ds.class_to_idx[cls]
        cls_indices = [i for i,t in enumerate(full_gan_ds.targets) if t == cls_idx]
        gan_ds      = torch.utils.data.Subset(full_gan_ds, cls_indices)
        gan_loader  = DataLoader(gan_ds, batch_size=BATCH_SIZE,
                                shuffle=True, num_workers=NUM_WORKERS)
        # -----------------------------------------------------------

        save_dir = f"generated_images_vqgan_{cls}"
        os.makedirs(save_dir, exist_ok=True)

        vqgan, disc = VQGAN().to(DEVICE), Discriminator().to(DEVICE)
        train_vqgan(vqgan, disc, gan_loader, EPOCHS_VQGAN, save_dir)

        # ---- draw + WRITE synthetic samples ------------
        n_synth    = math.ceil(len(gan_ds) * AUGMENT_MULTIPLIER)
        synth_imgs = sample(vqgan, n_synth)

        # FID gate:  skip bad GANs
        fid = quick_fid(os.path.join(DATA_ROOT, "train"), synth_imgs, cls)
        print(f"FID for {cls}: {fid:.1f}")

        # make sure gan_data root exists
        gan_data_root = "gan_data"
        os.makedirs(gan_data_root, exist_ok=True)

        if fid > FID_CUTOFF:
            print(f" -> FID worse than {FID_CUTOFF}; skipping synthetic images for {cls}.")
            synth_imgs = torch.empty(0, 3, IMG_HEIGHT, WIDTH)
        else:
            # save into class‐specific subfolder in gan_data/
            class_out = os.path.join(gan_data_root, cls)
            os.makedirs(class_out, exist_ok=True)
            for i, img in enumerate(synth_imgs):
                # each `img` is [3,H,W], in [0,1]
                save_image(img, os.path.join(class_out, f"{cls}_{i:05}.png"))

        n_synth = synth_imgs.size(0)
        if n_synth > 0:
            synth_lbls = torch.full((n_synth,), cls_idx, dtype=torch.long)
            synth_sets.append(TensorDataset(synth_imgs, synth_lbls))
        # ---------------------------------------------------

        del vqgan, disc
        torch.cuda.empty_cache()



    # 5-D concat all synthetic sets with the real training data
    aug_train_set = ConcatDataset([train_base, *synth_sets])

    # ----- NEW sampler  ----------------------------------------
    real_len   = len(train_base)
    synth_len  = sum(len(s) for s in synth_sets)
    weights    = [REAL_SYNTH_RATIO/real_len] * real_len + \
                [(1-REAL_SYNTH_RATIO)/synth_len] * synth_len
    sampler    = torch.utils.data.WeightedRandomSampler(
                    weights, num_samples=real_len + synth_len, replacement=True)
    aug_loader = DataLoader(aug_train_set,
                            batch_size=BATCH_SIZE,
                            sampler=sampler,
                            num_workers=NUM_WORKERS)
    # -----------------------------------------------------------


    # 5-E  train CNN AGAIN (data-augmented)
    cls_aug = SimpleCNN(len(class_names)).to(DEVICE)
    print("\nTraining CNN on original + synthetic images …")
    train_classifier(cls_aug, aug_loader, val_loader, EPOCHS_CLASSIFIER)
    acc_aug = evaluate(cls_aug, test_loader)
    print(f"Test accuracy (augmented): {acc_aug:.3%}")
