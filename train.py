import torch 
from torch.utils.data import DataLoader
import numpy as np 
import argparse
import yaml
from tqdm import tqdm 
import os 
from glob import glob
import gc

from model.generator import Generator
from model.discriminator import Discriminator
from utils.r1_loss import generator_loss, discriminator_loss
from dataset import Anime
from utils.history import History

class Trainer: 

    def __init__(self, hp):

        self.device = torch.device('cuda')
        self.generator = Generator(
                                res = hp['generator']['resolution'],
                                latent_size = hp['generator']['latent_size'],
                                deep_mapping = hp['generator']['deep_mapping']
                                ).to(self.device)
        self.optim_G = getattr(torch.optim, hp['generator']['optim'])(
                                params = self.generator.parameters(),
                                alpha = hp['generator']['alpha'], 
                                lr = hp['generator']['learning_rate']
                                )
        if hp['generator']['loss'] == 'r1': 
            self.generator_loss = generator_loss
        self.z_dim = hp['generator']['latent_size']

        self.discriminator = Discriminator(hp['discriminator']['resolution']).to(self.device)
        self.optim_D = getattr(torch.optim, hp['discriminator']['optim'])(
                                params = self.discriminator.parameters(),
                                alpha = hp['discriminator']['alpha'], 
                                lr = hp['discriminator']['learning_rate']
                                )
        if hp['discriminator']['loss'] == 'r1': 
            self.discriminator_loss = discriminator_loss
        self.lambda_gp = hp['discriminator']['lambda_gp']
        
        anime = Anime(hp['dataset_dir'], hp['image_size'])
        self.dataloader = DataLoader(anime, batch_size=hp['batch_size'], shuffle=True, pin_memory=False, drop_last=True)

        self.start_epoch = 0
        self.epochs = hp['epochs']
        self.batch_size = hp['batch_size']
        self.history = History()
        self.weight_dir = hp['weight_dir']
        self.Z = torch.randn((hp['generate_no_image']*hp['generate_no_image'], self.z_dim), device=self.device)
        if os.path.exists(self.weight_dir): 
            self.load_model()
        else: 
            os.mkdir(hp['weight_dir'])
        self.images_dir = hp['images_dir']
        if not os.path.exists(self.images_dir):
            os.mkdir(self.images_dir)
        self.save_frequency = hp['save_frequency']
        self.images_dir = hp['images_dir']
        self.generate_no_image = hp['generate_no_image']


    def generate_images(self, count=1):
        z = torch.randn((count, self.z_dim), device=self.device)
        return self.generator(z)

    def train_generator(self, fake_images): 
        logits_fake = self.discriminator(fake_images)
        return self.generator_loss(logits_fake)

    def train_discriminator(self, real_images, fake_images): 
        return self.discriminator_loss(self.discriminator, real_images, fake_images, self.lambda_gp)

    def train(self): 

        print("Training StyleGAN2")
        print("Generator : ")
        print(self.generator)
        print("Discriminator : ")
        print(self.discriminator)

        for epoch in range(self.start_epoch, self.epochs): 
            
            g_losses, d_losses = list(), list()
            for real_images in tqdm(self.dataloader):

                self.generator.zero_grad()
                real_images = real_images.to(self.device)
                fake_images = self.generate_images(self.batch_size).detach()

                d_loss = self.train_discriminator(real_images, fake_images)
                self.discriminator.zero_grad()
                d_loss.backward()
                self.optim_D.step()
                d_losses.append(d_loss)

                fake_images = self.generate_images(self.batch_size)
                g_loss = self.train_generator(fake_images)
                self.generator.zero_grad()
                g_loss.backward()
                self.optim_G.step()
                g_losses.append(g_loss)

            self.history.save_history(epoch, sum(g_losses)/len(g_losses), sum(d_losses)/len(d_losses))
            if epoch%self.save_frequency==0:
                self.save_model(epoch)
                self.history.plot_images(self.generator(self.Z).cpu(), epoch, self.images_dir, self.generate_no_image)
            gc.collect()
            torch.cuda.empty_cache()
        
        self.history.plot_loss(self.images_dir)
        


    def save_model(self, epoch): 

        pretrained_models = glob(os.path.join(self.weight_dir, 'model_*.pt'))
        if len(pretrained_models) >= 5:
            earliest_model = min(pretrained_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            os.remove(earliest_model)
        torch.save(self.generator.state_dict(), os.path.join(self.weight_dir, f'model_{epoch}.pt'))
        training_state = dict()
        training_state['generator'] = self.generator.state_dict()
        training_state['optim_G'] = self.optim_G.state_dict()
        training_state['discriminator'] = self.discriminator.state_dict()
        training_state['optim_D'] = self.optim_D.state_dict()
        training_state['epoch'] = epoch
        training_state['history'] = self.history
        training_state['Z'] = self.Z
        torch.save(training_state, os.path.join(self.weight_dir, 'training_state.pt'))

    def load_model(self): 

        if not os.path.exists(os.path.join(self.weight_dir, 'training_state.pt')):
            return 

        print("Resuming training from previous epoch")
        training_state = torch.load(os.path.join(self.weight_dir, 'training_state.pt'))
        self.generator.load_state_dict(training_state['generator'])
        self.optim_G.load_state_dict(training_state['optim_G'])
        self.discriminator.load_state_dict(training_state['discriminator'])
        self.optim_D.load_state_dict(training_state['optim_D'])
        self.start_epoch = training_state['epoch']+1
        self.history = training_state['history']
        self.Z = training_state['Z']


if __name__ == '__main__': 

    parser = argparse.ArgumentParser()
    parser.add_argument('--hyperparameters', type=str, default='hyperparameters.yaml', help='The path for hyperparameters.yaml file')
    args = parser.parse_args()

    hyperparameters = yaml.safe_load(open(args.hyperparameters, 'r'))
    trainer = Trainer(hyperparameters)
    trainer.train()