from model import ArcFace
from data_loader import build_data_loader
from utils import count_indentity, reduce_counter, reduce_indentity_dict, get_renumbering_identities_dict, split_dataset
import torch.nn as nn
import torch
from torch import optim
from tqdm import tqdm
from tensorboardX import SummaryWriter
import numpy as np
import os
import matplotlib.pyplot as plt

def cor_matrix_to_plt_image(matrix_tensor, step, apply_diagonal_zero=True):
    
    if apply_diagonal_zero:
        for i in range(len(matrix_tensor)):
            matrix_tensor[i, i] = 0
    
    fig, axes = plt.subplots(1, 3, figsize=(36, 12))

    axes[0].set_title(f'Speaker Embedding Correlation #{step:05d}', fontsize=24)
    im = axes[0].imshow(matrix_tensor, cmap='Spectral')
    fig.colorbar(im, ax=axes[0])

    axes[1].set_title(f'Normalized Correlation #{step:05d}', fontsize=24)
    im = axes[1].imshow(matrix_tensor, cmap='Spectral')
    im.set_clim([-1, 1])
    fig.colorbar(im, ax=axes[1])

    axes[2].hist(matrix_tensor.numpy().flatten(), 
             bins=np.arange(-1, 1, 0.05), 
             alpha = 0.5, density=True)

    axes[2].set_title(f'Correlation Distribution #{step:05d}', fontsize=24)

    fig.canvas.draw()

    image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    image_array = np.swapaxes(image_array, 0, 2)
    image_array = np.swapaxes(image_array, 1, 2)

    plt.close()

    return image_array

if __name__ == "__main__":

    identity_file = 'CelebA/Anno/identity_CelebA.txt'

    identity_counter, indentity_dict = count_indentity(identity_file)

    identity_counter = reduce_counter(identity_counter, 20)

    old_to_new_id_dict = get_renumbering_identities_dict(identity_counter)

    reduced_indentity_dict = reduce_indentity_dict(identity_counter, indentity_dict)

    dataset = [(key, old_to_new_id_dict[reduced_indentity_dict[key]]) for key in reduced_indentity_dict]

    train_dataset, test_dataset = split_dataset(dataset)

    reduced_number_speakers = len(old_to_new_id_dict)

    print(f'Reduced Indentity {reduced_number_speakers}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ArcFace(reduced_number_speakers).to(device)

    loss_function = nn.NLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_loader = build_data_loader(train_dataset)
    test_loader = build_data_loader(test_dataset)
    
    step = 1

    logging_step = 200

    summary_writer = SummaryWriter()
    summary_writer_eval = SummaryWriter(os.path.join(summary_writer.logdir, 'eval'))

    loss_list = list()
    acc_list = list()

    epoch = 10

    # print(model.state_dict().keys())
    v = model.state_dict()['identity_embedding.weight_v'].detach().cpu()
    v_norm = torch.norm(v, dim=1, keepdim=True)
    n = v / v_norm

    print(n.shape, v.shape)
    
    cor_mat = torch.matmul(n, n.T) # (H, W) * (W, H)
    print(torch.max(cor_mat), torch.min(cor_mat))

    matrix_image = cor_matrix_to_plt_image(cor_mat, step)
    summary_writer.add_image('identity_correlation', matrix_image, step)

    for i in range(epoch):

        model.train()
        for image, speaker in tqdm(train_loader):
            optimizer.zero_grad()
            # print(image)
            pred = model(image.to(device))
            loss = loss_function(pred, speaker.to(device))
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            acc = torch.mean((torch.argmax(pred.cpu(), dim=-1) == speaker).float())
            acc_list.append(acc)
            # print(loss.item())

            if step % logging_step == 0:
                summary_writer.add_scalar('loss', np.mean(loss_list), step)
                summary_writer.add_scalar('acc', np.mean(acc_list), step)
                loss_list = list()
                acc_list = list()

                v = model.state_dict()['identity_embedding.weight_v'].detach().cpu()
                v_norm = torch.norm(v, dim=1, keepdim=True)
                n = v / v_norm
                
                cor_mat = torch.matmul(n, n.T) # (H, W) * (W, H)
                print(torch.max(cor_mat), torch.min(cor_mat))

                matrix_image = cor_matrix_to_plt_image(cor_mat, step)
                summary_writer.add_image('identity_correlation', matrix_image, step)

            step += 1
            
        model.eval()
        loss_list = list()
        acc_list = list()
        with torch.no_grad():
            for image, speaker in tqdm(test_loader):
                pred = model(image.to(device))
                loss = loss_function(pred, speaker.to(device))
                loss_list.append(loss.item())
                acc = torch.mean((torch.argmax(pred.cpu(), dim=-1) == speaker).float())
                acc_list.append(acc)
        
        summary_writer_eval.add_scalar('loss', np.mean(loss_list), step)
        summary_writer_eval.add_scalar('acc', np.mean(acc_list), step)
        
