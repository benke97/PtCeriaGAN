import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from size_dataset import size_dataset_exp
from models import UNet_size_estimator
from tqdm import tqdm
import random
from math import inf
import pickle as pkl
from torch.nn.modules.loss import MSELoss, L1Loss

random.seed(1337)
"""Trains a U-Net to estimate size of a nanoparticle"""

# Prepare dataset, read in data and corresponding particle size
BATCH_SIZE = 32 #was 128
data_dir_train = "data/CycleGAN/train/"
data_dir_val = "data/CycleGAN/val/"
domain_names = ["sim","noisy"]
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_dataset = size_dataset_exp(data_dir_train, domain_names, transform)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = size_dataset_exp(data_dir_val, domain_names, transform)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet_size_estimator(1,1,ngf=64).to(device)

with open("data/exp_info.pkl", "rb") as f:
    exp_info = pkl.load(f)

unique_particle_ids = exp_info.unique_particle_id.unique()

class MSLELoss(nn.Module):
    def __init__(self):
        super(MSLELoss, self).__init__()

    def forward(self, y_pred, y_true):
        # Adding a small constant to ensure the logarithm is computable
        # when the inputs contain zeros
        epsilon = 1e-10
        # Compute the mean squared logarithmic error
        loss = torch.mean((torch.log(y_pred + 1 + epsilon) - torch.log(y_true + 1 + epsilon)) ** 2)
        return loss
    
class MAPELoss(nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, y_pred, y_true):
        epsilon = 1e-10  # To avoid division by zero
        loss = torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon)))
        return loss

criterion = L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.0008) 

num_epochs = 250
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
best_loss = inf
best_mean_percentage_error = inf
losses = []
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    loader_train = tqdm(train_dataloader, total=len(train_dataloader))

    for i, (clean_image, noisy_image, particle_size) in enumerate(loader_train):
        clean_image, noisy_image, particle_size = clean_image.to(device), noisy_image.to(device), particle_size.to(device)
        optimizer.zero_grad()
        outputs = model(clean_image)

        loss = criterion(outputs, particle_size.float()/1000)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        loader_train.set_postfix(loss=running_loss/(i+1))


    model.eval()
    running_loss = 0.0

    loader_val = tqdm(val_dataloader, total=len(val_dataloader))
    percentage_errors = []

    for i, (clean_image, noisy_image, particle_size) in enumerate(loader_val):
        clean_image, noisy_image, particle_size = clean_image.to(device), noisy_image.to(device), particle_size.to(device)
        outputs = model(clean_image)
        loss = criterion(outputs, particle_size.float()/1000)   
        running_loss += loss.item()

        # Calculate mean percentage error
        percentage_errors.append(torch.abs(outputs*1000 - particle_size).cpu().detach().numpy() / particle_size.cpu().detach().numpy())

        loader_val.set_postfix(loss=running_loss/(i+1))
    
    percentage_errors = np.concatenate(percentage_errors)
    mean_percentage_error = np.mean(percentage_errors)
    losses.append(running_loss/len(val_dataloader))
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(val_dataloader)}, Mean percentage error: {mean_percentage_error}")


    # Test case
    # Plot with matplotlib image, real size, predicted size
    if running_loss < best_loss:
        best_loss = running_loss
        best_mean_percentage_error = mean_percentage_error
        clean_image, noisy_image, particle_size = next(iter(val_dataloader))
        clean_image, noisy_image, particle_size = clean_image.to(device), noisy_image.to(device), particle_size.to(device) / 1000
        #outputs = model(noisy_image)
        #print(outputs, particle_size)
        #plt.figure()
        #plt.imshow(clean_image[0][0].cpu().detach().numpy(), cmap="gray")
        #plt.title(f"Predicted size: {outputs[0].item()*1000}, Real size: {particle_size[0].item()*1000}")
        #plt.show()
        #save model in directory "best_size_estimator"
        if not os.path.exists("best_size_estimator"):
            os.makedirs("best_size_estimator")
        torch.save(model.state_dict(), "best_size_estimator/size_estimator_sim_L1.pth")
        print("Model saved")
        #print("Predicted size: ", outputs[0].item(), "Real size: ", particle_size[0].item())
    
    lr_scheduler.step()

print("Minimum mean percentage error: ", best_mean_percentage_error)

#save losses
with open("size_estimator_sim_losses_L1.pkl", "wb") as f:
    pkl.dump(losses, f)
