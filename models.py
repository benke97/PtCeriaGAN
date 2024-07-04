import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

class SNPatchGANDiscriminator(nn.Module):
    def __init__(self, norm):
        super(SNPatchGANDiscriminator, self).__init__()

        def build_discriminator_block(in_channels, out_channels, norm_layer, use_bias):
            layers = [
                spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=use_bias)),
                nn.LeakyReLU(0.2)
            ]
            if norm_layer is not None:
                layers.append(norm_layer(out_channels))
            return nn.Sequential(*layers)

        norm_layer = get_norm_layer(norm)
        use_bias = norm_layer == nn.InstanceNorm2d

        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2)
        )

        self.conv2 = build_discriminator_block(64, 128, norm_layer, use_bias)
        self.conv3 = build_discriminator_block(128, 256, norm_layer, use_bias)
        self.conv4 = build_discriminator_block(256, 512, norm_layer, use_bias)
        self.output = spectral_norm(nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, img):
        x = self.conv1(img)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.output(x)
        return torch.sigmoid(x)

def get_norm_layer(norm):
    if norm == "instance":
        return nn.InstanceNorm2d
    elif norm == "batch":
        return nn.BatchNorm2d
    elif norm == "None":
        return None
    else:
        raise NotImplementedError("Normalization layer is not found:" + norm)
    

class UNet128_4_IN(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=16):
        super(UNet128_4_IN, self).__init__()
        
        # Downsample
        self.e1 = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1)
        self.e2 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True),
                                nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1),
                                nn.InstanceNorm2d(ngf * 2))
        self.e3 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True),
                                nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
                                nn.InstanceNorm2d(ngf * 4))
        self.e4 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True),
                                nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1),
                                nn.InstanceNorm2d(ngf * 8))
        self.e5 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True),
                                nn.Conv2d(ngf * 8, ngf * 16, kernel_size=4, stride=2, padding=1),
                                nn.InstanceNorm2d(ngf * 16))

        # Upsample

        self.d3 = nn.Sequential(nn.ReLU(inplace=True),
                                nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1),
                                nn.InstanceNorm2d(ngf * 8))
        self.d4 = nn.Sequential(nn.ReLU(inplace=True),
                                nn.ConvTranspose2d(ngf * 16, ngf * 4, kernel_size=4, stride=2, padding=1),
                                nn.InstanceNorm2d(ngf * 4))
        self.d5 = nn.Sequential(nn.ReLU(inplace=True),
                                nn.ConvTranspose2d(ngf * 8, ngf * 2, kernel_size=4, stride=2, padding=1),
                                nn.InstanceNorm2d(ngf * 2))
        self.d6 = nn.Sequential(nn.ReLU(inplace=True),
                                nn.ConvTranspose2d(ngf * 4, ngf, kernel_size=4, stride=2, padding=1),
                                nn.InstanceNorm2d(ngf))
        self.d7 = nn.Sequential(nn.ReLU(inplace=True),
                                nn.ConvTranspose2d(ngf * 2, output_nc, kernel_size=4, stride=2, padding=1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1) 
        e3 = self.e3(e2) 
        e4 = self.e4(e3) 
        e5 = self.e5(e4) 

        d3_ = self.d3(e5)
        d3 = torch.cat([d3_, e4], dim=1)
        d4_ = self.d4(d3) 
        d4 = torch.cat([d4_, e3], dim=1) 
        d5_ = self.d5(d4) 
        d5 = torch.cat([d5_, e2], dim=1) 
        d6_ = self.d6(d5) 
        d6 = torch.cat([d6_, e1], dim=1) 
        d7 = self.d7(d6) 
        
        o1 = self.sigmoid(d7)
        
        return o1
    
    def freeze_encoder(self):
        for param in self.e1.parameters():
            param.requires_grad = True
        for param in self.e2.parameters():
            param.requires_grad = True
        for param in self.e3.parameters():
            param.requires_grad = False
        for param in self.e4.parameters():
            param.requires_grad = False
        for param in self.e5.parameters():
            param.requires_grad = False
    

class UNet_size_estimator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=16):
        super(UNet_size_estimator, self).__init__()
        
        
        self.e1 = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1)
        self.e2 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True),
                                nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 2))
        self.e3 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True),
                                nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 4))
        self.e4 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True),
                                nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 8),
                                nn.Dropout(0.5))
        self.e5 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True),
                                nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 8),
                                nn.Dropout(0.5))


        self.d3 = nn.Sequential(nn.ReLU(inplace=True),
                                nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 8),
                                nn.Dropout(0.5))
        self.d4 = nn.Sequential(nn.ReLU(inplace=True),
                                nn.ConvTranspose2d(ngf * 16, ngf * 4, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 4),
                                nn.Dropout(0.5))
        self.d5 = nn.Sequential(nn.ReLU(inplace=True),
                                nn.ConvTranspose2d(ngf * 8, ngf * 2, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 2))
        self.d6 = nn.Sequential(nn.ReLU(inplace=True),
                                nn.ConvTranspose2d(ngf * 4, ngf, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf))
        self.d7 = nn.Sequential(nn.ReLU(inplace=True),
                                nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1))
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.fc = nn.Linear(ngf, 1)  
        self.softplus = nn.Softplus()  
        self.output = nn.ReLU()  


    def forward(self, x):

        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)

        d3_ = self.d3(e5) 
        d3 = torch.cat([d3_, e4], dim=1)
        d4_ = self.d4(d3)
        d4 = torch.cat([d4_, e3], dim=1)
        d5_ = self.d5(d4)
        d5 = torch.cat([d5_, e2], dim=1)
        d6_ = self.d6(d5)
        d6 = torch.cat([d6_, e1], dim=1)
        d7 = self.d7(d6)
        
        # Output
        pooled = self.pool(d7)
        flattened = torch.flatten(pooled, 1)
        o1 = self.fc(flattened).squeeze(-1)
        return o1