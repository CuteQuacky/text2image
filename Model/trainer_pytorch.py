from __future__ import print_function
from six.moves import range
from PIL import Image
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import numpy as np
import torchfile
import torchvision.utils as vutils
from PIL import Image, ImageDraw, ImageFont
import torch.nn.functional as F
from miscc.config import cfg
from miscc.utils import mkdir_p, weights_init, save_img_results, save_model, KL_loss, compute_discriminator_loss, compute_generator_loss
from tensorboard import summary
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
from scipy.stats import entropy
import torch.nn.parallel

class GANTrainer(object):
    def __init__(self, output_dir):
        # Directories for saving models, images, and logs
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            self.log_dir = os.path.join(output_dir, 'Log')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)
            mkdir_p(self.log_dir)
            self.summary_writer = SummaryWriter(self.log_dir)

        # Training parameters
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        # GPU settings
        s_gpus = cfg.GPU_ID
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        print(f"Number of GPUs: {self.num_gpus}")
        self.batch_size = cfg.TRAIN.BATCH_SIZE * self.num_gpus
        cudnn.benchmark = True
        self.device = torch.device(f"cuda:{self.gpus[0]}" if cfg.CUDA else "cpu")
        print(self.device)
        
    def load_network_stageI(self):
        #from model_pytorch import STAGE1G_CNN, STAGE1D_CNN
        
        #from model_pytorch import STAGE1G_ResBlock, STAGE1D_ResBlock
        
        from Model.model_pytorch import STAGE1_G_SE, STAGE1_D_SE
        
        # Initialize generator and discriminator
        netG = STAGE1_G_SE()
        netG.apply(weights_init)
        print(netG)
        netD = STAGE1_D_SE()
        netD.apply(weights_init)
        print(netD)

        if cfg.CUDA:
            netG = torch.nn.DataParallel(netG, device_ids=self.gpus).to(self.device)
            netD = torch.nn.DataParallel(netD, device_ids=self.gpus).to(self.device)
        return netG, netD

    def remove_module_prefix(self, state_dict):
        return {k.replace("module.", ""): v for k, v in state_dict.items()}
        
    def load_network_stageII(self):
        from Model.model_pytorch import STAGE1G_ResBlock, STAGE2G_ResBlock, STAGE2D_ResBlock

        # Initialize stage 1 and stage 2 generator
        Stage1_G = STAGE1G_ResBlock()
        netG = STAGE2G_ResBlock(Stage1_G)
        netG.apply(weights_init)
        print(netG)

        # Load pretrained Stage 1 generator if available
        if cfg.STAGE1_G != '':
            state_dict = torch.load(cfg.STAGE1_G, map_location=lambda storage, loc: storage)
            state_dict = self.remove_module_prefix(state_dict)
            netG.STAGE1_G.load_state_dict(state_dict)
            print('Load from: ', cfg.STAGE1_G)

        # Initialize stage 2 discriminator
        netD = STAGE2D_ResBlock()
        netD.apply(weights_init)
        print(netD)

        if cfg.CUDA:
            netG = torch.nn.DataParallel(netG, device_ids=self.gpus).to(self.device)
            netD = torch.nn.DataParallel(netD, device_ids=self.gpus).to(self.device)
        return netG, netD   

    def sample(self, datapath, stage=1):
        # Load network based on stage
        if stage == 1:
            netG, _ = self.load_network_stageI()
        else:
            netG, _ = self.load_network_stageII()
        netG.eval()

        # Load text embeddings
        t_file = torchfile.load(datapath)
        captions_list = t_file.raw_txt
        embeddings = np.concatenate(t_file.fea_txt, axis=0)
        num_embeddings = len(captions_list)
        print('Successfully load sentences from: ', datapath)
        print('Total number of sentences:', num_embeddings)
        print('num_embeddings:', num_embeddings, embeddings.shape)

        # Directory to save generated samples
        save_dir = cfg.NET_G[:cfg.NET_G.find('.pth')]
        mkdir_p(save_dir)

        # Generate samples
        batch_size = np.minimum(num_embeddings, self.batch_size)
        nz = cfg.Z_DIM
        device = torch.device("cuda:0" if cfg.CUDA else "cpu")
        noise = torch.FloatTensor(batch_size, nz).to(device)
        count = 0
        while count < num_embeddings:
            if count > 3000:
                break
            iend = count + batch_size
            if iend > num_embeddings:
                iend = num_embeddings
                count = num_embeddings - batch_size
            embeddings_batch = embeddings[count:iend]
            txt_embedding = torch.FloatTensor(embeddings_batch).to(device)

            # Generate fake images
            noise.normal_(0, 1)
            inputs = (txt_embedding, noise)
            _, fake_imgs, mu, logvar = netG(inputs)
            for i in range(batch_size):
                save_name = f'{save_dir}/{count + i}.png'
                im = fake_imgs[i].data.cpu().numpy()
                im = (im + 1.0) * 127.5
                im = im.astype(np.uint8)
                im = np.transpose(im, (1, 2, 0))
                im = Image.fromarray(im)
                im.save(save_name)
            count += batch_size

    def train(self, data_loader, stage=1):
        # Load network based on stage
        if stage == 1:
            netG, netD = self.load_network_stageI()
        else:
            netG, netD = self.load_network_stageII()

        # Training parameters
        nz = cfg.Z_DIM
        batch_size = self.batch_size
        noise = Variable(torch.FloatTensor(batch_size, nz)).to(self.device)
        fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1), volatile=True).to(self.device)
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1)).to(self.device)
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0)).to(self.device)

        generator_lr = cfg.TRAIN.GENERATOR_LR
        discriminator_lr = cfg.TRAIN.DISCRIMINATOR_LR
        lr_decay_step = cfg.TRAIN.LR_DECAY_EPOCH

        # Optimizers
        optimizerD = optim.Adam(netD.parameters(), lr=discriminator_lr, betas=(0.5, 0.999))
        netG_para = [p for p in netG.parameters() if p.requires_grad]
        optimizerG = optim.Adam(netG_para, lr=generator_lr, betas=(0.5, 0.999))

        count = 0
        for epoch in range(self.max_epoch):
            start_t = time.time()

            # Learning rate decay
            if epoch % lr_decay_step == 0 and epoch > 0:
                generator_lr *= 0.5
                for param_group in optimizerG.param_groups:
                    param_group['lr'] = generator_lr
                discriminator_lr *= 0.5
                for param_group in optimizerD.param_groups:
                    param_group['lr'] = discriminator_lr

            for i, data in enumerate(data_loader, 0):
                # Prepare training data
                real_img_cpu, txt_embedding = data
                real_imgs = Variable(real_img_cpu).to(self.device)
                txt_embedding = Variable(txt_embedding).to(self.device)

                # Generate fake images
                noise.data.normal_(0, 1)
                inputs = (txt_embedding, noise)
                _, fake_imgs, mu, logvar = netG(*inputs)

                # Update D network
                netD.zero_grad()
                errD, errD_real, errD_wrong, errD_fake = compute_discriminator_loss(netD, real_imgs, fake_imgs, real_labels, fake_labels, mu)
                errD.backward()
                optimizerD.step()

                # Update G network
                netG.zero_grad()
                errG = compute_generator_loss(netD, fake_imgs, real_labels, mu)
                kl_loss = KL_loss(mu, logvar)
                errG_total = errG + kl_loss * cfg.TRAIN.COEFF.KL
                errG_total.backward()
                optimizerG.step()

                count += 1

                # Logging
                if i % 50 == 0:
                    self.summary_writer.add_scalar('D_loss', errD.item(), count)
                    self.summary_writer.add_scalar('D_loss_real', errD_real, count)
                    self.summary_writer.add_scalar('D_loss_wrong', errD_wrong, count)
                    self.summary_writer.add_scalar('D_loss_fake', errD_fake, count)
                    self.summary_writer.add_scalar('G_loss', errG.item(), count)
                    self.summary_writer.add_scalar('KL_loss', kl_loss.item(), count)

                    # Save image results for each epoch
                    lr_fake, fake, _, _ = netG(txt_embedding, fixed_noise)
                    save_img_results(real_img_cpu, fake, epoch, self.image_dir)
                    if lr_fake is not None:
                        save_img_results(None, lr_fake, epoch, self.image_dir)

            end_t = time.time()
            print(f'''[{epoch}/{self.max_epoch}][{i}/{len(data_loader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} Loss_KL: {kl_loss.item():.4f}
                     Loss_real: {errD_real:.4f} Loss_wrong: {errD_wrong:.4f} Loss_fake {errD_fake:.4f}
                     Total Time: {end_t - start_t:.2f}sec
                  ''')
            if epoch % self.snapshot_interval == 0:
                save_model(netG, netD, epoch, self.model_dir)

        # Save final model
        save_model(netG, netD, self.max_epoch, self.model_dir)
        self.summary_writer.close()