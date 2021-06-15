import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms
from utils import prepare_cifar
from fgsm_attack import FGSMAttack
from tqdm import tqdm, trange
from pgd20 import pgd20_attack
from model import get_model_for_attack


def eval_model(model, test_loader, device):
    correct_adv, correct = [], []
    distance = []
    num = 0
    with trange(10000) as pbar:
        for x, label in test_loader:
            x, label = x.to(device), label.to(device)
            batch, c, h, w = x.shape
            model.eval()
            with torch.no_grad():
                output = model(x)
            pred = output.argmax(dim=1)
            correct.append(pred == label)
            num += x.shape[0]
            pbar.set_description(f"Acc: {torch.cat(correct).float().mean():.5f}")
            pbar.update(x.shape[0])
    natural_acc = torch.cat(correct).float().mean()
    return natural_acc, distance


def eval_model_pgd(model, test_loader, device, step_size, epsilon, perturb_steps):
    correct_adv, correct = [], []
    distance = []
    num = 0
    with trange(10000) as pbar:
        for x, label in test_loader:
            x, label = x.to(device), label.to(device)
            batch, c, h, w = x.shape
            x_adv = pgd20_attack(model, x.clone(), label.clone(), step_size, epsilon, perturb_steps)
            x_adv = x_adv.to(device)
            model.eval()
            with torch.no_grad():
                output = model(x)
                output_adv = model(x_adv)
            distance.append(torch.max((x - x_adv).reshape(batch, -1).abs(), dim=1)[0])
            pred = output.argmax(dim=1)
            pred_adv = output_adv.argmax(dim=1)
            correct.append(pred == label)
            correct_adv.append(pred_adv == label)
            num += x.shape[0]
            pbar.set_description(
                f"Acc: {torch.cat(correct).float().mean():.5f}, Robust Acc:{torch.cat(correct_adv).float().mean():.5f}")
            pbar.update(x.shape[0])
    natural_acc = torch.cat(correct).float().mean()
    robust_acc = torch.cat(correct_adv).float().mean()
    distance = torch.cat(distance).max()
    return natural_acc, robust_acc, distance


def eval_model_with_attack(model, test_loader, attack, epsilon, device):
    correct_adv, correct = [], []
    distance = []
    num = 0
    nb = 0
    with trange(10000) as pbar:
        for x, label in test_loader:
            x, label = x.to(device), label.to(device)
            batch, c, h, w = x.shape
            # x_adv = attack(x.clone(), label.clone())
            x_adv = attack(model, x.clone(), label.clone())
            # x_adv = attack.perturb(x)
            # x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
            x_adv = x_adv.clamp(0, 1)
            x_adv = x_adv.to(device)
            model.eval()
            with torch.no_grad():
                output = model(x)
                output_adv = model(x_adv)
            distance.append(torch.max((x - x_adv).reshape(batch, -1).abs(), dim=1)[0])
            pred = output.argmax(dim=1)
            pred_adv = output_adv.argmax(dim=1)
            correct.append(pred == label)
            correct_adv.append(pred_adv == label)
            num += x.shape[0]
            nb += 1
            pbar.set_description(
                f"Acc: {torch.cat(correct).float().mean():.5f}, Robust Acc:{torch.cat(correct_adv).float().mean():.5f}")
            pbar.update(x.shape[0])
    natural_acc = torch.cat(correct).float().mean()
    robust_acc = torch.cat(correct_adv).float().mean()
    distance = torch.cat(distance).max()
    return natural_acc, robust_acc, distance
