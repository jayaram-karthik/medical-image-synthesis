import numpy as np
import torch
import torch.nn as nn

from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torchvision import transforms


from utils import *
from generator import Generator
from discriminator import Discriminator

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# hyperparameters
lr = 0.0002
beta_1 = 0.5 
beta_2 = 0.999

z_dim = 100
batch_size = 128

fixed_noise = get_noise(batch_size, z_dim, device=device)

train_transform = transforms.Compose([
    transforms.ToTensor(),
])

dataloader = ImageFolder(
    'data/',
    batch_size=batch_size,
    transform=train_transform
)

generator = Generator(z_dim).to(device)
generator_opt = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta_1, beta_2))

discriminator  = Discriminator().to(device) 
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta_1, beta_2))

generator = generator.apply(weights_init)
discriminator = discriminator.apply(weights_init)   

n_epochs = 10
cur_step = 0
total_steps = 0
cur_step = 0

c_lambda = 10
discriminator_repeats = 5
display_step = 50

for epoch in range(n_epochs):
    cur_step = 0
    for real, _ in dataloader:
        cur_batch_size = len(real)
        real = real.to(device)

        for _ in range(discriminator_repeats):

            discriminator_optimizer.zero_grad()
            fake_noise = get_noise(cur_batch_size, z_dim, device=device)
            fake = generator(fake_noise)
            discriminator_fake_pred = discriminator(fake.detach())
            discriminator_real_pred = discriminator(real)

            epsilon = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)
            gradient = get_gradient(discriminator, real, fake.detach(), epsilon)
            gp = gradient_penalty(gradient)
            discriminator_loss = get_discriminator_loss(discriminator_fake_pred, discriminator_real_pred, gp, c_lambda)

            # Update gradients
            discriminator_loss.backward(retain_graph=True)
            # Update optimizer
            discriminator_optimizer.step()

        generator_opt.zero_grad()
        fake_noise_2 = get_noise(cur_batch_size, z_dim, device=device)
        fake_2 = generator(fake_noise_2)
        discriminator_fake_pred = discriminator(fake_2)
        
        generator_loss = get_generator_loss(discriminator_fake_pred)
        generator_loss.backward()

        # Update the weights
        generator_opt.step()

        cur_step += 1
        total_steps += 1
    print()
    print(f"Epoch: {epoch}/{n_epochs} Total Steps:{total_steps}")
    print(f"Loss_G : {generator_loss}")
    
    fake_noise = fixed_noise
    fake = generator(fake_noise)
    
    show_tensor_images(fake, show_fig=True,epoch=epoch)
    
    cur_step = 0

torch.save(generator)