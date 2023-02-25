import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os


class History:

    def __init__(self): 

        self.generator_loss = list()
        self.discriminator_loss = list()
        self.epochs = list()

    def save_history(self, epoch, generator_loss, discriminator_loss): 

        self.generator_loss.append(generator_loss)
        self.discriminator_loss.append(discriminator_loss)
        self.epochs.append(epoch)

        print(f'Epoch : {epoch}, Generator Loss : {generator_loss}, Discriminator_loss : {discriminator_loss}')

    def plot_images(self, images, epoch, image_dir, generate_no_image): 

        grid = torchvision.utils.make_grid(images, nrow=generate_no_image, padding=5)
        np_grid = grid.permute(1, 2, 0).numpy()
        plt.imshow(np_grid)
        plt.axis('off')
        plt.savefig(os.path.join(image_dir, f'images_{epoch}.png'))
        plt.clf()

    def plot_loss(self, image_dir): 

        plt.style.use('seaborn-darkgrid')
        fig, ax = plt.subplots()
        ax.plot(self.epochs, self.generator_loss, label='Generator Loss')
        ax.plot(self.epochs, self.discriminator_loss, label='Discriminator Loss')
        ax.legend()
        ax.set(xlabel='Epochs', ylabel='Loss', title='GAN Loss Over Epochs')
        plt.tight_layout()
        plt.savefig(os.path.join(image_dir, f'loss.png'))


