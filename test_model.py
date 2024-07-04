import os
from models import UNet_size_estimator, UNet128_4_IN
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

data_dir = "example_series"
files = os.listdir(data_dir)

#----------------- Estimate size from a time-series -----------------#

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
size_estimator_checkpoint = "weights/trained_size_estimator.pth"
size_estimator = UNet_size_estimator(1,1,ngf=64).to(device)
size_estimator.load_state_dict(torch.load(size_estimator_checkpoint))
size_estimator.eval()

predictions = []
for file in files:
    img = np.load(os.path.join(data_dir, file))["arr_0"]   
    img = torch.tensor(img).float().unsqueeze(0).unsqueeze(0).to(device)
    prediction = size_estimator(img)
    predictions.append(prediction.item()*1000)

prediction = int(np.mean(predictions))
print(f"Estimated size of nanoparticle: {prediction} atoms")


#----------------- Denoising an experimental image series -----------------#

denoiser = UNet128_4_IN(1,1,ngf=32).to(device)
cycle_gan_checkpoint = "weights/trained_CycleGAN.pth"
denoiser.load_state_dict(torch.load(cycle_gan_checkpoint)["G_BA_state_dict"])
denoiser.eval()

denoised_images = []
images = []

for file in files:
    img = np.load(os.path.join(data_dir, file))["arr_0"]   
    img = torch.tensor(img).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        denoised_img = denoiser(img)
    denoised_images.append(denoised_img.squeeze().cpu().detach().numpy())
    images.append(img.squeeze().cpu().detach().numpy())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Initialize the images
img1 = ax1.imshow(images[0], cmap='gray')
img2 = ax2.imshow(denoised_images[0], cmap='gray')

ax1.set_title('Raw Image')
ax2.set_title('Denoised Image')
ax1.axis('off')
ax2.axis('off')
# Define an update function for the animation
def update(frame):
    img1.set_data(images[frame])
    img2.set_data(denoised_images[frame])
    return img1, img2

# Create the animation
ani = FuncAnimation(fig, update, frames=len(images), interval=200, blit=True)

plt.show()