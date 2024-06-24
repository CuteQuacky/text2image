import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import os
from scipy.stats import entropy
import pickle
import argparse

# Custom dataset to load the generated images
class GeneratedImagesDataset(Dataset):
    def __init__(self, fake_image_dir, filenames, transform=None):
        self.fake_image_dir = fake_image_dir
        self.filenames = filenames
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fake_img_name = os.path.join(self.fake_image_dir, f'fake_samples_{self.filenames[idx].replace("/", "-")}.png')
        fake_image = Image.open(fake_img_name).convert('RGB')
        
        if self.transform:
            fake_image = self.transform(fake_image)
            
        return fake_image

# Inception Score calculation function
def inception_score(imgs, cuda=True, batch_size=64, resize=False, splits=10):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)
    
    assert batch_size > 0
    assert N > batch_size

    # Set up dataloader
    dataloader = DataLoader(imgs, batch_size=batch_size)

    # Load Inception v3 model
    inception_model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False).eval()
    if cuda:
        inception_model = inception_model.cuda()

    up = transforms.Resize((299, 299), interpolation=transforms.InterpolationMode.BILINEAR)
    
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return torch.nn.functional.softmax(x, dim=1).data.cpu().numpy()

    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        if cuda:
            batch = batch.cuda()
        batchv = torch.autograd.Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for calculating Inception Score")
    parser.add_argument('--fake_image_dir', type=str, required=True, help="Directory where generated images are saved")
    parser.add_argument('--filenames_path', type=str, required=True, help="Path to the filenames file")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # Load filenames
    with open(args.filenames_path, 'rb') as f:
        filenames = pickle.load(f, encoding='latin1')

    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Create dataset and dataloader for fake images
    dataset_fake = GeneratedImagesDataset(args.fake_image_dir, filenames, transform=transform)

    # Calculate Inception Score for fake images
    mean_fake, std_fake = inception_score(dataset_fake, cuda=torch.cuda.is_available(), batch_size=32, resize=True, splits=10)
    print(f"Fake Images Inception Score: {mean_fake} ± {std_fake}")