import torch
import torch.nn as nn
from models import UNet128_4_IN, SNPatchGANDiscriminator
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from image_pool import ImagePool
from cycleGAN_dataset import CycleGAN_dataset
import torchvision.transforms
import torch.nn.functional as F
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
import time
from torchvision.transforms.functional import to_pil_image, to_tensor
import torch.autograd as autograd

random.seed(1337)

def save_checkpoint(epoch, G_AB, G_BA, D_A, D_B, optimizer_G, optimizer_D_A, optimizer_D_B, loss, checkpoint_dir="checkpoints"):
    
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    #save txt with all settings lambda_gp, lambda_cycle, etc
    with open(f"{checkpoint_dir}/settings.txt", "w") as f:
        f.write(f"lambda_gp: {lambda_gp}\n")
        f.write(f"lambda_cycle: {lambda_cycle}\n")
        f.write(f"lambda_id: {lambda_id}\n")
        f.write(f"initial_lr_G: {initial_lr_G}\n")
        f.write(f"initial_lr_D_A: {initial_lr_D_A}\n")
        f.write(f"initial_lr_D_B: {initial_lr_D_B}\n")
        f.write(f"num_epochs: {num_epochs}\n")
        f.write(f"norm: {norm}\n")
        f.write(f"G_AB: {G_AB}\n")
        f.write(f"G_BA: {G_BA}\n")
        f.write(f"ngf: {ngf}\n")
        f.write(f"D_A: {D_A}\n")
        f.write(f"D_B: {D_B}\n")
        f.write(f"best_fid: {best_fid}\n")
        f.write(f"checkpoint_freq: {checkpoint_freq}\n")
        f.write(f"buffer_D_B: {buffer_D_B}\n")
        f.write(f"buffer_D_A: {buffer_D_A}\n")
        f.write(f"date: {date}\n")
        f.write(f"loss: {loss}\n")
        f.write(f"with_FFT: {with_FFT}\n")
        f.write(f"freeze_endocers: {freeze_encoders}\n")
        f.write(f"lambda_gp_FFT: {lambda_gp_FFT}\n")
        f.write(f"add_noise: {add_noise}\n")
        f.write(f"batch_size: {BATCH_SIZE}\n")
    checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth"
    torch.save({
        'epoch': epoch,
        'G_AB_state_dict': G_AB.state_dict(),
        'G_BA_state_dict': G_BA.state_dict(),
        'D_A_state_dict': D_A.state_dict(),
        'D_B_state_dict': D_B.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_A_state_dict': optimizer_D_A.state_dict(),
        'optimizer_D_B_state_dict': optimizer_D_B.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN-GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=real_samples.device)
    
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    
    # Forward pass of interpolated samples through the discriminator
    d_interpolates = D(interpolates)
    
    # Create labels to compute the gradients of the outputs w.r.t. the inputs
    fake = torch.ones(d_interpolates.size(), device=real_samples.device, requires_grad=False)
    
    # Compute gradients w.r.t. the interpolated samples
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Reshape the gradients to [batch_size, -1]
    gradients = gradients.view(gradients.size(0), -1)
    
    # Compute the gradient penalty
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty

def prepare_images_for_fid(image_list):
    processed_images = []
    for img_tensor in image_list:
        # Convert each tensor to PIL Image, then back to tensor to ensure correct format
        # This also rescales images to [0, 255] and converts them to uint8
        for img in img_tensor:
            pil_img = to_pil_image(img.cpu())  # Convert to PIL Image to rescale correctly
            processed_img = to_tensor(pil_img).to(device) * 255  # Back to tensor and scale
            processed_images.append(processed_img)
    # Stack all images into a single tensor
    return torch.stack(processed_images).type(torch.uint8)

checkpoint_freq = 25
BATCH_SIZE = 4
data_dir_train = "data/CycleGAN/train/"
data_dir_val = "data/CycleGAN/val/"
domain_names = ["sim","exp"]
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_dataset = CycleGAN_dataset(data_dir_train, domain_names, transform)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = CycleGAN_dataset(data_dir_val, domain_names, transform)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

def adjust_learning_rate_generator(optimizer, epoch, initial_lr, num_epochs):
    """ Adjusts learning rate each epoch: constant for first half, linear decay in second half. """
    if epoch < num_epochs / 2: #was num_epochs / 2:
        lr = initial_lr
    else:
        lr = initial_lr * float(num_epochs - epoch) / (num_epochs / 2)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate_discriminator(optimizer, epoch, initial_lr, num_epochs):
    """ Adjusts learning rate each epoch: constant for first half, linear decay in second half. """
    if epoch < num_epochs / 2: #was num_epochs / 2:
        lr = initial_lr
    else:
        lr = initial_lr * float(num_epochs - epoch) / (num_epochs / 2)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def make_3_channel_tensor(tensor):
    return torch.cat((tensor, tensor, tensor), 1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
with_FFT = False
norm = "instance"

ngf = 32
G_AB = UNet128_4_IN(1,1,ngf=ngf).to(device)
G_BA = UNet128_4_IN(1,1,ngf=ngf).to(device)

D_A = SNPatchGANDiscriminator(norm).to(device)
D_B = SNPatchGANDiscriminator(norm).to(device)
if with_FFT:
    D_A_FFT = SNPatchGANDiscriminator(norm).to(device)
    D_B_FFT = SNPatchGANDiscriminator(norm).to(device)


checkpoint = torch.load("weights/pretrained_weights.pth") 

G_AB.load_state_dict(checkpoint["G_AB_state_dict"])
G_BA.load_state_dict(checkpoint["G_BA_state_dict"])

freeze_encoders = True
if freeze_encoders:
    G_AB.freeze_encoder()
    G_BA.freeze_encoder()

criterion_GAN = nn.BCELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

initial_lr_G = 1e-5 
initial_lr_D_A = 1e-5 
initial_lr_D_B = 1e-5 

optimizer_G = torch.optim.Adam(list(G_AB.parameters()) + list(G_BA.parameters()), lr=initial_lr_G, betas=(0.5, 0.999))

if with_FFT:
    optimizer_D_A = torch.optim.Adam(list(D_A.parameters()) + list(D_A_FFT.parameters()), lr=initial_lr_D_A, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(list(D_B.parameters()) + list(D_B_FFT.parameters()), lr=initial_lr_D_B, betas=(0.5, 0.999))
else:
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=initial_lr_D_A, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=initial_lr_D_B, betas=(0.5, 0.999))

lambda_cycle = 300 
lambda_id = 0.25 * lambda_cycle 
lambda_gp = 0 
lambda_gp_FFT = 1

num_epochs = 500
buffer_D_B = ImagePool(20) 
buffer_D_A = ImagePool(20)

if with_FFT:    
    buffer_D_A_FFT = ImagePool(20)
    buffer_D_B_FFT = ImagePool(20)

lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze', normalize=True).to(device) #was squeeze

fixed_sample = next(iter(val_dataloader))
fixed_sample_2 = next(iter(val_dataloader))

#select 3 exp images and 3 simulated images, these will be selected by hand from data/noiser/sim and data/noiser/exp
#each epoch we plot two 2x3 images, one with the 3 exp images plus G_AB(sim) and the other with the 3 sim images plus G_BA(exp)
exp_files = ["data/CycleGAN/val/exp/205_7.npz","data/CycleGAN/val/exp/753_29.npz","data/CycleGAN/val/exp/7427_219.npz"]
sim_files = ["data/CycleGAN/val/sim/179.npz","data/CycleGAN/val/sim/2851.npz","data/CycleGAN/val/sim/4871.npz"]
exp_images = []
sim_images = []
for file in exp_files:
    img = np.load(file)["arr_0"]
    img = torch.unsqueeze(transform(to_pil_image(img)),0).to(device)
    exp_images.append(img)

for file in sim_files:
    img = np.load(file)["arr_0"]
    img = torch.unsqueeze(transform(to_pil_image(img)),0).to(device)
    sim_images.append(img)

best_fid = 1000
add_noise = False
writer = SummaryWriter()
batch_id = 0

date = time.strftime("%Y-%m-%d_%H-%M-%S")

def add_gaussian_noise(image, mean=0, std=0.0025):
    noise = torch.randn(image.size()).to(image.device) * std + mean
    noisy_image = torch.clamp(image + noise, 0, 1)
    return noisy_image

for epoch in range(num_epochs):
    torch.cuda.empty_cache()
    print(f"Epoch {epoch+1}/{num_epochs}")
    adjust_learning_rate_generator(optimizer_G, epoch, initial_lr_G, num_epochs)
    adjust_learning_rate_discriminator(optimizer_D_A, epoch, initial_lr_D_A, num_epochs)
    adjust_learning_rate_discriminator(optimizer_D_B, epoch, initial_lr_D_B, num_epochs)

    G_AB.train()
    G_BA.train()
    D_A.train()
    D_B.train()
    train_loss_G = 0
    train_loss_D_A = 0
    train_loss_D_B = 0

    loader_train = tqdm(train_dataloader, total=len(train_dataloader))

    #Training
    for real_A, real_B in loader_train:
        #sim, exp
        optimizer_G.zero_grad()

        assert torch.max(real_B) <= 1
        assert torch.min(real_B) >= 0

        real_A = real_A.to(device)
        real_B = real_B.to(device)

        if with_FFT:
            real_A_FFT = torch.abs(torch.fft.fftshift(torch.fft.fft2(real_A)))
            real_B_FFT = torch.abs(torch.fft.fftshift(torch.fft.fft2(real_B)))
        
        if add_noise:
            real_A_noisy = add_gaussian_noise(real_A)
            fake_B = G_AB(real_A_noisy)
        else:
            fake_B = G_AB(real_A)

        if with_FFT:
            fake_B_FFT = torch.abs(torch.fft.fftshift(torch.fft.fft2(fake_B)))

        discriminator_output = D_B(fake_B)
        
        if with_FFT:
            discriminator_output_FFT = D_B_FFT(fake_B_FFT)
        target_tensor = torch.ones_like(discriminator_output).to(device)
        if with_FFT:
            loss_GAN_AB = criterion_GAN(discriminator_output, target_tensor) + criterion_GAN(discriminator_output_FFT, target_tensor)
        else:    
            loss_GAN_AB = criterion_GAN(discriminator_output, target_tensor)

        fake_A = G_BA(real_B)
        if with_FFT:
            fake_A_FFT = torch.abs(torch.fft.fftshift(torch.fft.fft2(fake_A)))

        discriminator_output = D_A(fake_A)
        if with_FFT:
            discriminator_output_FFT = D_A_FFT(fake_A_FFT)
        target_tensor = torch.ones_like(discriminator_output).to(device)
        if with_FFT:
            loss_GAN_BA = criterion_GAN(discriminator_output, target_tensor) + criterion_GAN(discriminator_output_FFT, target_tensor)
        else:
            loss_GAN_BA = criterion_GAN(discriminator_output, target_tensor)

        recovered_A = G_BA(fake_B)
        loss_cycle_A = criterion_cycle(recovered_A, real_A)

        if add_noise:
            fake_A_noisy = add_gaussian_noise(fake_A)
            recovered_B = G_AB(fake_A_noisy)
        else:
            recovered_B = G_AB(fake_A)

        loss_cycle_B = lpips(make_3_channel_tensor(recovered_B), make_3_channel_tensor(real_B))

        loss_id_A = criterion_identity(G_BA(real_A), real_A)
        loss_id_B = lpips(make_3_channel_tensor(G_AB(real_B)), make_3_channel_tensor(real_B))

        loss_G = loss_GAN_AB + loss_GAN_BA + lambda_cycle * (loss_cycle_B + loss_cycle_A) + lambda_id * (loss_id_A + loss_id_B)

        loss_G.backward()
        optimizer_G.step()


        optimizer_D_A.zero_grad()
        
        real_A_preds = D_A(real_A)

        if with_FFT:
            real_A_preds_FFT = D_A_FFT(real_A_FFT)

        target_real = torch.ones_like(real_A_preds).to(device)

        if with_FFT:
            loss_real = criterion_GAN(real_A_preds, target_real) + criterion_GAN(real_A_preds_FFT, target_real)
        else:
            loss_real = criterion_GAN(real_A_preds, target_real)

        fake_A_buffered = buffer_D_A.query(fake_A).to(device)
        fake_A_preds = D_A(fake_A.detach())

        if with_FFT:
            fake_A_FFT_buffered = buffer_D_A_FFT.query(fake_A_FFT).to(device)
            fake_A_preds_FFT = D_A_FFT(fake_A_FFT.detach())

        target_fake = torch.zeros_like(fake_A_preds).to(device)

        if with_FFT:
            loss_fake = criterion_GAN(fake_A_preds, target_fake) + criterion_GAN(fake_A_preds_FFT, target_fake)
        else:
            loss_fake = criterion_GAN(fake_A_preds, target_fake)

        gradient_penalty = compute_gradient_penalty(D_A, real_A.data, fake_A.data)

        if with_FFT:
            gradient_penalty_FFT = compute_gradient_penalty(D_A_FFT, real_A_FFT.data, fake_A_FFT.data)

        if with_FFT:
            loss_D_A = (loss_real + loss_fake) / 2 + lambda_gp * gradient_penalty + lambda_gp_FFT * gradient_penalty_FFT
            writer.add_scalar('batch_loss/FFT_D_A', lambda_gp_FFT * gradient_penalty_FFT, batch_id)
        else:
            loss_D_A = (loss_real + loss_fake) / 2 + lambda_gp * gradient_penalty

        loss_D_A.backward()  
        optimizer_D_A.step()

        optimizer_D_B.zero_grad()

        real_B_preds = D_B(real_B)
        
        if with_FFT:
            real_B_preds_FFT = D_B_FFT(real_B_FFT)
        
        target_real = torch.ones_like(real_B_preds).to(device)
        
        if with_FFT:
            loss_real = criterion_GAN(real_B_preds, target_real) + criterion_GAN(real_B_preds_FFT, target_real)
        else:
            loss_real = criterion_GAN(real_B_preds, target_real)

        fake_B_buffered = buffer_D_B.query(fake_B).to(device)
        fake_B_preds = D_B(fake_B_buffered.detach())
        
        if with_FFT:
            fake_B_FFT_buffered = buffer_D_B_FFT.query(fake_B_FFT).to(device)
            fake_B_preds_FFT = D_B_FFT(fake_B_FFT_buffered.detach())
        
        target_fake = torch.zeros_like(fake_B_preds).to(device)
        
        if with_FFT:
            loss_fake = criterion_GAN(fake_B_preds, target_fake) + criterion_GAN(fake_B_preds_FFT, target_fake)
        else:
            loss_fake = criterion_GAN(fake_B_preds, target_fake)

        gradient_penalty = compute_gradient_penalty(D_B, real_B.data, fake_B.data)
        
        if with_FFT:
            gradient_penalty_FFT = compute_gradient_penalty(D_B_FFT, real_B_FFT.data, fake_B_FFT.data)

        if with_FFT:
            loss_D_B = (loss_real + loss_fake) / 2 + lambda_gp * gradient_penalty   + lambda_gp_FFT * gradient_penalty_FFT
            writer.add_scalar('batch_loss/FFT_D_B', lambda_gp_FFT * gradient_penalty_FFT, batch_id)
        else:
            loss_D_B = (loss_real + loss_fake) / 2 + lambda_gp * gradient_penalty
    
        loss_D_B.backward()
        optimizer_D_B.step()

        train_loss_G += loss_G.item()
        train_loss_D_A += loss_D_A.item()
        train_loss_D_B += loss_D_B.item()

        writer.add_scalar('batch_loss/cycle_sim', lambda_cycle * loss_cycle_A.item(), batch_id)
        writer.add_scalar('batch_loss/cycle_exp', lambda_cycle * loss_cycle_B.item(), batch_id)
        writer.add_scalar('batch_loss/id_sim', lambda_id * loss_id_A.item(), batch_id)
        writer.add_scalar('batch_loss/id_exp', lambda_id * loss_id_B.item(), batch_id)
        writer.add_scalar('batch_loss/GAN_AB', loss_GAN_AB.item(), batch_id)
        writer.add_scalar('batch_loss/GAN_BA', loss_GAN_BA.item(), batch_id)
        writer.add_scalar('batch_loss/loss_D_A', loss_D_A.item(), batch_id)
        writer.add_scalar('batch_loss/loss_D_B', loss_D_B.item(), batch_id)
        
        batch_id += 1

    writer.add_scalar('Loss/train_G', train_loss_G/len(train_dataloader), epoch)
    writer.add_scalar('Loss/train_D_A', train_loss_D_A/len(train_dataloader), epoch)
    writer.add_scalar('Loss/train_D_B', train_loss_D_B/len(train_dataloader), epoch)
    writer.add_scalar('Learning rate/G', optimizer_G.param_groups[0]['lr'], epoch)
    writer.add_scalar('Learning rate/D_A', optimizer_D_A.param_groups[0]['lr'], epoch)
    writer.add_scalar('Learning rate/D_B', optimizer_D_B.param_groups[0]['lr'], epoch)

    print(f"Loss G: {train_loss_G/len(train_dataloader)}, Loss D_A: {train_loss_D_A/len(train_dataloader)}, Loss D_B: {train_loss_D_B/len(train_dataloader)}")

    G_AB.eval()
    G_BA.eval()
    D_A.eval()
    D_B.eval()
    
    val_loss_G = 0
    val_loss_D_A = 0
    val_loss_D_B = 0

    loader_val = tqdm(val_dataloader, total=len(val_dataloader))
    
    #Validation
    real_A_list = []
    real_B_list = []
    fake_A_list = []
    fake_B_list = []    
    with torch.no_grad():
        for real_A, real_B in loader_val:
            #sim,exp
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            fake_B = G_AB(real_A)
        
            discriminator_output = D_B(fake_B)
            target_tensor = torch.ones_like(discriminator_output).to(device)  
            loss_GAN_AB = criterion_GAN(discriminator_output, target_tensor)

            fake_A = G_BA(real_B)

            discriminator_output = D_A(fake_A)
            target_tensor = torch.ones_like(discriminator_output).to(device)

            loss_GAN_BA = criterion_GAN(discriminator_output, target_tensor)

            recovered_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recovered_A, real_A)
            recovered_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recovered_B, real_B)

            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)

            loss_G = loss_GAN_AB + loss_GAN_BA + lambda_cycle * (loss_cycle_B + loss_cycle_A) + lambda_id * (loss_id_A + loss_id_B)

            val_loss_G += loss_G.item()

            real_A_preds = D_A(real_A)
            target_real = torch.ones_like(real_A_preds).to(device)
            loss_real = criterion_GAN(real_A_preds, target_real)

            fake_A_preds = D_A(fake_A)
            target_fake = torch.zeros_like(fake_A_preds).to(device)
            loss_fake = criterion_GAN(fake_A_preds, target_fake)

            loss_D_A = (loss_real + loss_fake) / 2 
            val_loss_D_A += loss_D_A.item()

            real_B_preds = D_B(real_B)
            target_real = torch.ones_like(real_B_preds).to(device)
            loss_real = criterion_GAN(real_B_preds, target_real)

            fake_B_preds = D_B(fake_B)
            target_fake = torch.zeros_like(fake_B_preds).to(device)
            loss_fake = criterion_GAN(fake_B_preds, target_fake)

            loss_D_B = (loss_real + loss_fake) / 2
            val_loss_D_B += loss_D_B.item()

            for i in range(real_A.size(0)):
                real_A_list.append(make_3_channel_tensor(real_A[i].unsqueeze(0)))
                real_B_list.append(make_3_channel_tensor(real_B[i].unsqueeze(0)))
                fake_A_list.append(make_3_channel_tensor(fake_A[i].unsqueeze(0)))
                fake_B_list.append(make_3_channel_tensor(fake_B[i].unsqueeze(0)))

    writer.add_scalar('Loss/val_G', val_loss_G/len(val_dataloader), epoch)
    writer.add_scalar('Loss/val_D_A', val_loss_D_A/len(val_dataloader), epoch)
    writer.add_scalar('Loss/val_D_B', val_loss_D_B/len(val_dataloader), epoch)

    print(f"Val Loss G: {val_loss_G/len(val_dataloader)}, Val Loss D_A: {val_loss_D_A/len(val_dataloader)}, Val Loss D_B: {val_loss_D_B/len(val_dataloader)}")

    #Visualization tensorboard
    with torch.no_grad():
        fixed_sample_A = fixed_sample[0][0].unsqueeze(0).to(device)
        fixed_sample_B = fixed_sample[1][0].unsqueeze(0).to(device)
        fixed_fake_B = G_AB(fixed_sample_A)
        fixed_fake_A = G_BA(fixed_sample_B)
        fixed_recovered_A = G_BA(fixed_fake_B)
        fixed_recovered_B = G_AB(fixed_fake_A)
        
        def resize_img(img):
            return F.interpolate(img, size=(512, 512), mode='nearest')

        fixed_sample_A_resized = resize_img(fixed_sample_A)
        fixed_fake_B_resized = resize_img(fixed_fake_B)
        fixed_recovered_A_resized = resize_img(fixed_recovered_A)
        fixed_sample_B_resized = resize_img(fixed_sample_B)
        fixed_fake_A_resized = resize_img(fixed_fake_A)
        fixed_recovered_B_resized = resize_img(fixed_recovered_B)

        fixed_cycle_ABA = torch.cat([fixed_sample_A_resized, fixed_fake_B_resized, fixed_recovered_A_resized], dim=3)
        fixed_cycle_BAB = torch.cat([fixed_sample_B_resized, fixed_fake_A_resized, fixed_recovered_B_resized], dim=3)

        writer.add_image('ABA_cycle', fixed_cycle_ABA.squeeze(0), epoch)
        writer.add_image('BAB_cycle', fixed_cycle_BAB.squeeze(0), epoch)

        fixed_sample_2_A = fixed_sample_2[0][0].unsqueeze(0).to(device)
        fixed_sample_2_B = fixed_sample_2[1][0].unsqueeze(0).to(device)
        fixed_fake_B_2 = G_AB(fixed_sample_2_A)
        fixed_fake_A_2 = G_BA(fixed_sample_2_B)
        fixed_recovered_A_2 = G_BA(fixed_fake_B_2)
        fixed_recovered_B_2 = G_AB(fixed_fake_A_2)

        fixed_sample_2_A_resized = resize_img(fixed_sample_2_A)
        fixed_fake_B_2_resized = resize_img(fixed_fake_B_2)
        fixed_recovered_A_2_resized = resize_img(fixed_recovered_A_2)
        fixed_sample_2_B_resized = resize_img(fixed_sample_2_B)
        fixed_fake_A_2_resized = resize_img(fixed_fake_A_2)
        fixed_recovered_B_2_resized = resize_img(fixed_recovered_B_2)

        fixed_cycle_ABA_2 = torch.cat([fixed_sample_2_A_resized, fixed_fake_B_2_resized, fixed_recovered_A_2_resized], dim=3)
        fixed_cycle_BAB_2 = torch.cat([fixed_sample_2_B_resized, fixed_fake_A_2_resized, fixed_recovered_B_2_resized], dim=3)

        writer.add_image('ABA_cycle_2', fixed_cycle_ABA_2.squeeze(0), epoch)
        writer.add_image('BAB_cycle_2', fixed_cycle_BAB_2.squeeze(0), epoch)
    
    #Visualization of 3 exp and 3 sim images
    with torch.no_grad():
        # get the fake versions of the 3 exp images and 3 sim images
        fake_exp_images = []
        fake_sim_images = []
        
        for exp_img in exp_images:
            fake_sim_images.append(G_BA(exp_img))
        for sim_img in sim_images:
            fake_exp_images.append(G_AB(sim_img))
        
        plt.subplot(2, 3, 1)
        plt.imshow(np.squeeze(exp_images[0].cpu().numpy()), cmap='gray')
        plt.axis('off')
        plt.subplot(2, 3, 2)
        plt.imshow(np.squeeze(exp_images[1].cpu().numpy()), cmap='gray')
        plt.axis('off')
        plt.subplot(2, 3, 3)
        plt.imshow(np.squeeze(exp_images[2].cpu().numpy()), cmap='gray')
        plt.axis('off')
        plt.subplot(2, 3, 4)
        plt.imshow(np.squeeze(fake_exp_images[0].cpu().numpy()), cmap='gray')
        plt.axis('off')
        plt.subplot(2, 3, 5)
        plt.imshow(np.squeeze(fake_exp_images[1].cpu().numpy()), cmap='gray')
        plt.axis('off')
        plt.subplot(2, 3, 6)
        plt.imshow(np.squeeze(fake_exp_images[2].cpu().numpy()), cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"epoch_figs/exp_images_{epoch}.png")
        plt.close()

        plt.subplot(2, 3, 1)
        plt.imshow(np.squeeze(sim_images[0].cpu().numpy()), cmap='gray')
        plt.axis('off')
        plt.subplot(2, 3, 2)
        plt.imshow(np.squeeze(sim_images[1].cpu().numpy()), cmap='gray')
        plt.axis('off')
        plt.subplot(2, 3, 3)
        plt.imshow(np.squeeze(sim_images[2].cpu().numpy()), cmap='gray')
        plt.axis('off')
        plt.subplot(2, 3, 4)
        plt.imshow(np.squeeze(fake_sim_images[0].cpu().numpy()), cmap='gray')
        plt.axis('off')
        plt.subplot(2, 3, 5)
        plt.imshow(np.squeeze(fake_sim_images[1].cpu().numpy()), cmap='gray')
        plt.axis('off')
        plt.subplot(2, 3, 6)
        plt.imshow(np.squeeze(fake_sim_images[2].cpu().numpy()), cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"epoch_figs/sim_images_{epoch}.png")
        plt.close()

        #resize and cat images
        exp_images_resized = []
        fake_exp_images_resized = []

        for img in exp_images:
            exp_images_resized.append(resize_img(img))
        for img in fake_exp_images:
            fake_exp_images_resized.append(resize_img(img))

        sim_images_resized = []
        fake_sim_images_resized = []

        for img in sim_images:
            sim_images_resized.append(resize_img(img))
        for img in fake_sim_images:
            fake_sim_images_resized.append(resize_img(img))

        # Concatenate images for cycle visualization in a 2x3 grid
        cycle_exp = torch.cat([exp_images_resized[0], exp_images_resized[1], exp_images_resized[2], fake_exp_images_resized[0], fake_exp_images_resized[1], fake_exp_images_resized[2]], dim=3)
        cycle_sim = torch.cat([sim_images_resized[0], sim_images_resized[1], sim_images_resized[2], fake_sim_images_resized[0], fake_sim_images_resized[1], fake_sim_images_resized[2]], dim=3)

        # Log images
        writer.add_image('cycle_exp', cycle_exp.squeeze(0), epoch)
        writer.add_image('cycle_sim', cycle_sim.squeeze(0), epoch)

    #FID calculation
    fid_A = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    fid_B = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    # Stack lists into tensors for FID calculation
    real_A_tensor = torch.cat(real_A_list, dim=0)
    real_B_tensor = torch.cat(real_B_list, dim=0)
    fake_A_tensor = torch.cat(fake_A_list, dim=0)
    fake_B_tensor = torch.cat(fake_B_list, dim=0)

    # Update FID calculators
    fid_A.update(real_A_tensor, real=True)
    fid_A.update(fake_A_tensor, real=False)
    fid_B.update(real_B_tensor, real=True)
    fid_B.update(fake_B_tensor, real=False)

    # Compute FID scores
    fid_A_score = fid_A.compute().item()
    fid_B_score = fid_B.compute().item()

    # Log FID scores
    writer.add_scalar('FID/FID_A', fid_A_score, epoch)
    writer.add_scalar('FID/FID_B', fid_B_score, epoch)

    # Calculate and log the average FID score
    tot_fid = (fid_A_score + fid_B_score) / 2
    writer.add_scalar('FID/FID', tot_fid, epoch)

    # Reset FID calculators
    fid_A.reset()
    fid_B.reset()

    # Save the best model based on FID score
    if tot_fid < best_fid:
        best_fid = tot_fid
        save_checkpoint(epoch, G_AB, G_BA, D_A, D_B, optimizer_G, optimizer_D_A, optimizer_D_B, 
                        [train_loss_G/len(train_dataloader), train_loss_D_A/len(train_dataloader), train_loss_D_B/len(train_dataloader), 
                        val_loss_G/len(val_dataloader), val_loss_D_A/len(val_dataloader), val_loss_D_B/len(val_dataloader)],
                        checkpoint_dir="best_fid_checkpoint")

    #Save checkpoint
    if (epoch+1) % checkpoint_freq == 0:
        save_loss = [train_loss_G/len(train_dataloader), train_loss_D_A/len(train_dataloader), train_loss_D_B/len(train_dataloader), val_loss_G/len(val_dataloader), val_loss_D_A/len(val_dataloader), val_loss_D_B/len(val_dataloader)]
        save_checkpoint(epoch, G_AB, G_BA, D_A, D_B, optimizer_G, optimizer_D_A, optimizer_D_B, save_loss,checkpoint_dir="exp_checkpoints/"+date)