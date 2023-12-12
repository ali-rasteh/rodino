import os

from advertorch.attacks import LinfPGDAttack, L2PGDAttack
from tqdm import tqdm
import torch.nn as nn
from torchvision import transforms as pth_transforms, datasets
import vision_transformer as vits
import utils

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import torch


def plot_histogram(data_list, label_list, save_path, color_list=["#008080", "#FCA592", "yellow", "lightblue"],
                   y_bins_max=80, y_bins_slot=10, x_bins_max=100, x_bins_slot=10, label_size=20):
    sns.set(style="darkgrid")
    bins = np.arange(0, x_bins_max, x_bins_slot)
    ybins = np.arange(0, y_bins_max, y_bins_slot)
    plt.rcParams['font.size'] = 2

    fig, ax = plt.subplots(figsize=(10, 7))
    for idx, data in enumerate(data_list):
        sns.histplot(data=data, color=color_list[idx], label=label_list[idx], kde=True, bins=100)

    plt.xlabel("")
    plt.ylabel("")
    plt.legend(fontsize=22)
    plt.tick_params(axis='both', which='major', labelsize=label_size)
    ax.set_xticks(bins)
    ax.set_yticks(ybins)
    ax.set(xlim=(0, x_bins_max), ylim=(0, y_bins_max))
    plt.savefig(save_path, format="pdf", dpi=300)
    plt.show()


def generate_attack(img_ref, target_model, eps, attack='linf'):
    if attack == 'linf':
        adversary = LinfPGDAttack(target_model, loss_fn=nn.MSELoss(), eps=eps, nb_iter=50,
                                  eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1., targeted=False)
    else:
        adversary = L2PGDAttack(target_model, loss_fn=nn.MSELoss(), eps=eps, nb_iter=100,
                                eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1., targeted=False)
    img_ref = adversary(img_ref, target_model(img_ref))

    return img_ref


def distance_attack_eval(eps, dataloader, model, name, attack):
    distance_list = list()
    print(f'Attack to {name} model...')
    count = 0
    for idx, data in tqdm(enumerate(dataloader)):
        inputs = data[0].cuda()
        input_embed = model(inputs).detach()
        adv_inputs = generate_attack(img_ref=inputs, target_model=model, eps=eps, attack=attack)

        adv_input_embed = model(adv_inputs).detach()
        if attack == 'linf':
            dist = torch.norm(input_embed - adv_input_embed, p=float('inf'), dim=1).cpu().tolist()
        else:
            dist = torch.norm(input_embed - adv_input_embed, p=2, dim=1).cpu().tolist()

        distance_list.extend(dist)
        count += inputs.shape[0]
        if count > 1000:
            break
    torch.save(distance_list, f'{name}_distance_list_pgd_{eps}')
    return distance_list


def get_model(name):
    arch = 'vit_small'
    model = vits.__dict__[arch](patch_size=16, num_classes=0)
    utils.load_pretrained_weights(model, f'./save/{name}/checkpoint.pth', 'teacher', arch, 16)
    model.eval()
    model = model.cuda()
    return model


if __name__ == '__main__':
    val_transform = pth_transforms.Compose([
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
    ])
    dataset_val = datasets.ImageFolder(os.path.join('/data/sara/imagenet-100/', "val"), transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=64,
        num_workers=4,
        pin_memory=True,
    )
    orig_name = 'orig'
    adv_name = 'adv'
    eps = 1.0
    attack = 'l2'
    model = get_model(adv_name)
    adv_distance_list = distance_attack_eval(eps=eps, dataloader=val_loader, model=model, name=adv_name, attack=attack)
    model = get_model(orig_name)
    distance_list = distance_attack_eval(eps=eps, dataloader=val_loader, model=model, name=orig_name, attack=attack)
    # plot_histogram([torch.load('adv_distance_list_pgd_1.0'), torch.load('orig_distance_list_pgd_1.0')],
    #                label_list=['RoDINO', 'DINO'], save_path='./fig.pdf')
    plot_histogram([adv_distance_list, distance_list], label_list=['RoDINO', 'DINO'], save_path='./fig.pdf')