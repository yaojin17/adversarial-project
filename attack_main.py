import torch
import torch.nn as nn
import torch.nn.functional as F

# from torchvision import datasets, transforms
from utils import prepare_cifar, get_test_cifar
from EarlyEndPGD20_attack import EarlyEndPGDAttack
from fgsm_attack import FGSMAttack
from deepfool_attack import DeepFoolAttack
from models import WideResNet, WideResNet34, WideResNet28,ResNet18
from model import get_model_for_attack
from FWEEPGD20 import FWEEPGDAttack
from SemiTargetedPGD20 import SemiTargetedPGDAttack
from black_box_attack import BlackBoxAttack
from pgd20 import PGD20Attack
from tqdm import tqdm, trange
from ChangedResnet18 import ChangedResnet18
from eval_model import eval_model, eval_model_pgd, eval_model_with_attack
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Test Robust Accuracy')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--step_size', type=int, default=0.003,
                        help='step size for pgd attack(default:0.003)')
    parser.add_argument('--epsilon', type=float, default=8/255.0,
                        help='max distance for pgd attack (default epsilon=8/255)')
    parser.add_argument('--perturb_steps', type=int, default=20,
                        help='iterations for pgd attack (default pgd20)')
    parser.add_argument('--model_name', type=str, default="")
    parser.add_argument('--method_name', type=str, default="FWEEPGDAttack")
    parser.add_argument('--model_path', type=str, default="./weight/resnet18-e76-0.8400_0.5001-best.pt")
    parser.add_argument('--gpu_id', type=str, default="0")
    return parser.parse_args()


def get_attack_method(arg, method_name):
    if method_name == "FWEEPGDAttack":
        return FWEEPGDAttack(arg.step_size, arg.epsilon, arg.perturb_steps)
    elif method_name == "EarlyEndPGDAttack":
        return EarlyEndPGDAttack(arg.step_size, arg.epsilon, arg.perturb_steps)
    elif method_name == "FGSMAttack":
        return FGSMAttack(arg.step_size, arg.epsilon, arg.perturb_steps)
    elif method_name == "DeepFoolAttack":
        return DeepFoolAttack(arg.step_size, arg.epsilon, arg.perturb_steps)
    elif method_name == "SemiTargetedPGDAttack":
        return SemiTargetedPGDAttack(arg.step_size, arg.epsilon, arg.perturb_steps)
    elif method_name == "BlackBoxAttack":
        return BlackBoxAttack(arg.step_size, arg.epsilon, arg.perturb_steps)
    elif method_name == "PGD20Attack":
        return PGD20Attack(arg.step_size, arg.epsilon, arg.perturb_steps)
    else:
        print("The name of attack method is wrong")


if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id  # 多卡机设置使用的gpu卡号
    gpu_num = max(len(args.gpu_id.split(',')), 1)
    device = torch.device('cuda')
    if args.model_name != "":
        model = get_model_for_attack(args.model_name).to(device)  # 根据model_name, 切换要攻击的model
        model = nn.DataParallel(model, device_ids=[i for i in range(gpu_num)])

    else:
        # 防御任务, Change to your model here
        model = WideResNet28()
        model.load_state_dict(torch.load('weight/wide_resnet28-e91-0.8640_0.5832-best.pt'))
        model = nn.DataParallel(model, device_ids=[i for i in range(gpu_num)])
    # 攻击任务：Change to your attack function here
    # Here is a attack baseline: PGD attack
    attack = get_attack_method(args,args.method_name)

    model.eval()
    test_loader = get_test_cifar(args.batch_size)
    natural_acc, robust_acc, distance = eval_model_with_attack(model, test_loader, attack, args.epsilon, device)
    print(f"Natural Acc: {natural_acc:.5f}, Robust acc: {robust_acc:.5f}, distance:{distance:.5f}")
