import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import prepare_cifar, get_test_cifar
from models import WideResNet, WideResNet34, WideResNet28,ResNet18
from eval_model import eval_model
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Test Robust Accuracy')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--model_name', type=str, default="")
    parser.add_argument('--model_path', type=str, default="./weight/wide_resnet28-e91-0.8640_0.5832-best.pt")
    parser.add_argument('--gpu_id', type=str, default="0")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id  # 多卡机设置使用的gpu卡号
    gpu_num = max(len(args.gpu_id.split(',')), 1)
    device = torch.device('cuda')

    model = WideResNet28()
    model.load_state_dict(torch.load(args.model_path))
    model = nn.DataParallel(model, device_ids=[i for i in range(gpu_num)])

    model.eval()
    test_loader = get_test_cifar(args.batch_size)
    natural_acc,_ = eval_model(model,test_loader,device)
    print(f"Natural Acc: {natural_acc:.5f}")