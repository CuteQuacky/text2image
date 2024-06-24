import os
import errno
import numpy as np

from copy import deepcopy
from miscc.config import cfg

from torch.nn import init
import torch
import torch.nn as nn
import torchvision.utils as vutils



import pickle

def load_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f, encoding='latin1')

def decode_text(embedding, embedding_to_filename):
    embedding_tuple = tuple(embedding.flatten())
    
    filename = embedding_to_filename.get(embedding_tuple)
    print(f"Filename {filename}")
    if filename:
        text_file = f"text_descriptions/{filename}.txt"  # Adjust the path as necessary
        try:
            with open(text_file, 'r') as f:
                text_description = f.read().strip()
            return text_description
        except FileNotFoundError:
            return "Text description not found."
    else:
        return "Embedding not found in the pre-trained embeddings."



def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD


def compute_discriminator_loss(netD, real_imgs, fake_imgs,
                               real_labels, fake_labels,
                               conditions):
    criterion = nn.BCELoss()
    batch_size = real_imgs.size(0)
    cond = conditions.detach()
    fake = fake_imgs.detach()
    real_features = netD(real_imgs)
    fake_features = netD(fake)
    # real pairs
    inputs = (real_features, cond)
    real_logits = netD.module.get_cond_logits(*inputs)
    errD_real = criterion(real_logits, real_labels)
    # wrong pairs
    inputs = (real_features[:(batch_size-1)], cond[1:])
    wrong_logits = netD.module.get_cond_logits(*inputs)
    errD_wrong = criterion(wrong_logits, fake_labels[1:])
    # fake pairs
    inputs = (fake_features, cond)
    fake_logits = netD.module.get_cond_logits(*inputs)
    errD_fake = criterion(fake_logits, fake_labels)

    if netD.module.get_uncond_logits is not None:
        real_logits = netD.module.get_uncond_logits(real_features)
        fake_logits = netD.module.get_uncond_logits(fake_features)
        uncond_errD_real = criterion(real_logits, real_labels)
        uncond_errD_fake = criterion(fake_logits, fake_labels)
        errD = ((errD_real + uncond_errD_real) / 2. +
                (errD_fake + errD_wrong + uncond_errD_fake) / 3.)
        errD_real = (errD_real + uncond_errD_real) / 2.
        errD_fake = (errD_fake + uncond_errD_fake) / 2.
    else:
        errD = errD_real + (errD_fake + errD_wrong) * 0.5
    return errD, errD_real.item(), errD_wrong.item(), errD_fake.item()


def compute_generator_loss(netD, fake_imgs, real_labels, conditions):
    criterion = nn.BCELoss()
    cond = conditions.detach()
    fake_features = netD(fake_imgs)
    # fake pairs
    inputs = (fake_features, cond)
    fake_logits = netD.module.get_cond_logits(*inputs)
    errD_fake = criterion(fake_logits, real_labels)
    if netD.module.get_uncond_logits is not None:
        fake_logits = netD.module.get_uncond_logits(fake_features)
        uncond_errD_fake = criterion(fake_logits, real_labels)
        errD_fake += uncond_errD_fake
    return errD_fake


#############################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


#############################
def save_img_results(data_img, fake, epoch, image_dir):
    num = cfg.VIS_COUNT
    fake = fake[0:num]
    # data_img is changed to [0,1]
    if data_img is not None:
        data_img = data_img[0:num]
        vutils.save_image(
            data_img, '%s/real_samples_%03d.png' % (image_dir, epoch),
            normalize=True)
        # fake.data is still [-1, 1]
        vutils.save_image(
            fake.data, '%s/fake_samples_epoch_%03d.png' %
            (image_dir, epoch), normalize=True)
    else:
        vutils.save_image(
            fake.data, '%s/lr_fake_samples_epoch_%03d.png' %
            (image_dir, epoch), normalize=True)



def save_model(netG, netD, epoch, model_dir):
    netG_path = '%s/netG_epoch_%d.pth' % (model_dir, epoch)
    netD_path = '%s/netD_epoch_last.pth' % (model_dir)
    
    torch.save(netG.state_dict(), netG_path)
    torch.save(netD.state_dict(), netD_path)
    
    print('Save G/D models')
    print(f'Generator model saved to: {netG_path}')
    print(f'Discriminator model saved to: {netD_path}')


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
