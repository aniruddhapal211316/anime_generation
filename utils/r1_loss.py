import torch
import torch.nn.functional as F
from torch.autograd import grad

def r1loss(inputs, label=None):
    l = -1 if label else 1
    return F.softplus(l*inputs).mean()


def discriminator_loss(D, real_imgs, fake_imgs, lambda_gp):
    real_imgs.requires_grad = True
    real_outputs = D(real_imgs)
    d_real_loss = r1loss(real_outputs, True)
    grad_real = grad(outputs=real_outputs.sum(), inputs=real_imgs, create_graph=True)[0]
    grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
    grad_penalty = 0.5*lambda_gp*grad_penalty
    D_x_loss = d_real_loss + grad_penalty
    
    fake_logits = D(fake_imgs)
    D_z_loss = r1loss(fake_logits, False)
    D_loss = D_x_loss + D_z_loss
    return D_loss
    
    
def generator_loss(logits_fake):
    loss = r1loss(logits_fake, True)
    return loss
