import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid

import matplotlib.pyplot as plt

def get_noise(n_samples, z_dim, device='cpu'):
    return torch.randn(n_samples,z_dim,device=device)

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

def gradient_penalty(gradient):
    gradient = gradient.view(len(gradient), -1)

    gradient_norm = gradient.norm(2, dim=1)
    
    penalty = torch.mean((gradient_norm - 1)**2)
    return penalty

def get_generator_loss(discriminator_predictions):
    gen_loss = -1. * torch.mean(discriminator_predictions)
    return gen_loss

def get_discriminator_loss(discriminator_fake_predictions, discriminator_real_predictions, gp, c_lambda):
    crit_loss = torch.mean(discriminator_fake_predictions) - torch.mean(discriminator_real_predictions) + c_lambda * gp
    return crit_loss

def get_gradient(discriminator, real, fake, epsilon):

    mixed_images = real * epsilon + fake * (1 - epsilon)

    mixed_scores = discriminator(mixed_images)
    
    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores), 
        create_graph=True,
        retain_graph=True,    
    )[0]
    return gradient

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28), show_fig=False, epoch=0):
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.axis('off')
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    if show_fig:
        plt.savefig('image_epoch_{:04d}.png'.format(epoch))
        
    plt.show()