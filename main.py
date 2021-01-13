from model import ArcFace
from data_loader import build_data_loader
from utils import count_indentity, reduce_counter, reduce_indentity_dict, get_renumbering_identities_dict, split_dataset
import torch.nn as nn
import torch
from torch import optim
from tqdm import tqdm
from tensorboardX import SummaryWriter
import numpy as np

if __name__ == "__main__":

    identity_file = 'CelebA/Anno/identity_CelebA.txt'

    identity_counter, indentity_dict = count_indentity(identity_file)

    identity_counter = reduce_counter(identity_counter, 20)

    old_to_new_id_dict = get_renumbering_identities_dict(identity_counter)

    reduced_indentity_dict = reduce_indentity_dict(identity_counter, indentity_dict)

    dataset = [(key, old_to_new_id_dict[reduced_indentity_dict[key]]) for key in reduced_indentity_dict]

    train_dataset, test_dataset = split_dataset(dataset)

    reduced_number_speakers = len(reduced_indentity_dict)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ArcFace(reduced_number_speakers).to(device)

    loss_function = nn.NLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_loader = build_data_loader(train_dataset)
    test_loader = build_data_loader(test_dataset)
    
    step = 1

    logging_step = 1000

    summary_writer = SummaryWriter()

    loss_list = list()

    epoch = 10

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
            # print(loss.item())

            if step % logging_step == 0:
                summary_writer.add_scalar('train/loss', np.mean(loss_list), step)
                loss_list = list()

            step += 1
            
        model.eval()
        loss_list = list()
        for image, speaker in tqdm(test_loader):
            pred = model(image.to(device))
            loss = loss_function(pred, speaker.to(device))
            loss_list.append(loss.item())
        
        summary_writer.add_scalar('eval/loss', np.mean(loss_list), step)
        
