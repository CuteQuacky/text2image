import numpy as np
import pickle
import torch
import torchvision.utils as vutils
import os
from collections import OrderedDict
from torch.autograd import Variable
import argparse

# Load pre-trained embeddings
def load_pretrained_embeddings(embedding_path):
    with open(embedding_path, 'rb') as f:
        embeddings = pickle.load(f, encoding='latin1')
    embeddings = np.array(embeddings)  # Ensure embeddings are a single numpy array
    return embeddings

# Load class info
def load_class_info(class_info_path):
    with open(class_info_path, 'rb') as f:
        class_info = pickle.load(f, encoding='latin1')
    return class_info

# Load filenames
def load_filenames(filenames_path):
    with open(filenames_path, 'rb') as f:
        filenames = pickle.load(f, encoding='latin1')
    return filenames

# Load captions based on filenames
def load_captions(captions_dir, filenames):
    captions = []
    for filename in filenames:
        caption_file = os.path.join(captions_dir, filename + '.txt')
        with open(caption_file, 'r') as f:
            captions.append(f.read().split('\n'))
    return captions

# Save generated images
def save_img_results(fake, filenames, image_dir):
    os.makedirs(image_dir, exist_ok=True)
    for i in range(len(fake)):
        sanitized_filename = filenames[i].replace('/', '-')
        vutils.save_image(
            fake[i].data,
            os.path.join(image_dir, f'fake_samples_{sanitized_filename}.png'),
            normalize=True
        )

# Save captions
def save_captions(captions, filenames, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    captions_path = os.path.join(output_dir, "captions.txt")
    with open(captions_path, 'w') as f:
        for i, caption_list in enumerate(captions):
            f.write(f"{filenames[i]}:\n")
            for caption in caption_list:
                f.write(f"  {caption}\n")
            f.write("\n")

# Load the pre-trained generator
def load_network_stageI(model_path, device):
    from Model.model_pytorch import STAGE1G_CNN
    netG = STAGE1G_CNN() 
    state_dict = torch.load(model_path, map_location=device)

    # Remove 'module.' prefix
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '') 
        new_state_dict[name] = v
    
    netG.load_state_dict(new_state_dict)
    netG.to(device)
    netG.eval()
    return netG

def test(embeddings, captions, model_path, output_dir, device, filenames):
    netG = load_network_stageI(model_path, device)
    batch_size = 64
    nz = 100 

    num_embeddings = len(embeddings)
    count = 0

    while count < num_embeddings:
        iend = count + batch_size
        if iend > num_embeddings:
            iend = num_embeddings
        embeddings_batch = embeddings[count:iend]

        txt_embedding = torch.FloatTensor(np.mean(embeddings_batch, axis=1)).to(device)

        noise = Variable(torch.FloatTensor(txt_embedding.size(0), nz)).to(device)

        print(f"Processing batch from {count} to {iend}")
        print(f"Shape of txt_embedding: {txt_embedding.shape}")
        print(f"Shape of noise: {noise.shape}")

        noise.data.normal_(0, 1)
        inputs = (txt_embedding, noise)
        
        print(f"Shape of inputs[0] (txt_embedding): {inputs[0].shape}")
        print(f"Shape of inputs[1] (noise): {inputs[1].shape}")

        try:
            _, fake_imgs, mu, logvar = netG(*inputs)
        except RuntimeError as e:
            print(f"RuntimeError: {e}")
            return

        save_img_results(fake_imgs, filenames[count:iend], output_dir)
        count = iend

    save_captions(captions, filenames, output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for loading and testing the GAN model")
    parser.add_argument('--embedding_path', type=str, required=True, help="Path to the embeddings file")
    parser.add_argument('--class_info_path', type=str, required=True, help="Path to the class info file")
    parser.add_argument('--filenames_path', type=str, required=True, help="Path to the filenames file")
    parser.add_argument('--captions_dir', type=str, required=True, help="Directory containing text captions")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory for saving generated images")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # embedding_path = '/home/user/Sources/Python/txt2img/Datasets/birds/test/char-CNN-RNN-embeddings.pickle'
    # class_info_path = '/home/user/Sources/Python/txt2img/Datasets/birds/test/class_info.pickle'
    # filenames_path = '/home/user/Sources/Python/txt2img/Datasets/birds/test/filenames.pickle'
    
    # Load pretrained embeddings
    embeddings = load_pretrained_embeddings(args.embedding_path)
    # Load class information
    class_info = load_class_info(args.class_info_path)
    # Load filenames
    filenames = load_filenames(args.filenames_path)

    # Print some basic information
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Shape of the first embedding: {embeddings[0].shape}")
    print(f"Number of class info entries: {len(class_info)}")
    print(f"Number of filenames: {len(filenames)}")

    #captions_dir = '/home/user/Sources/Python/txt2img/Datasets/birds/text_c10'
    
    # Load captions corresponding to the filenames
    captions = load_captions(args.captions_dir, filenames)

    # Print some basic information about captions
    print(f"Number of captions: {len(captions)}")
    print(f"Example caption: {captions[0]}")

    # model_path = '/home/user/Sources/Python/output/birds__2024_06_20_20_28_55/Model/netG_epoch_300.pth' 
    # output_dir = '/home/user/Sources/Python/txt2img/test-output-cnn' 
    
    # Set device to GPU if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Run the test function with the provided arguments
    test(embeddings, captions, args.model_path, args.output_dir, device, filenames)