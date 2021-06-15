import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import WideResNet28
import time
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from models import ResNet18
from tqdm import tqdm, trange
from pgd20 import pgd20_attack
from attack_main import eval_model_pgd
from utils import prepare_cifar, Logger, check_mkdir
from eval_model import eval_model, eval_model_pgd
from trades_awp_utils import perturb_input,TradesAWP


def parse_args():
    parser = argparse.ArgumentParser(description='Test Robust Accuracy')
    parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test_batch_size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--step_size', type=int, default=0.003,
                        help='step size for pgd attack(default:0.03)')
    parser.add_argument('--perturb_steps', type=int, default=10,
                        help='iterations for pgd attack (default pgd10)')
    parser.add_argument('--epsilon', type=float, default=8. / 255.,
                        help='max distance for pgd attack (default epsilon=8/255)')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate')
    # parser.add_argument('--lr_steps', type=str, default=,
    #                help='iterations for pgd attack (default pgd20)')
    parser.add_argument('--epoch', type=int, default=200,
                        help='epochs for pgd training ')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='iterations for pgd attack (default pgd20)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay ratio')
    parser.add_argument('--adv_train', type=int, default=1,
                        help='If use adversarial training')
    # parser.add_argument('--model_path', type=str, default="./models/model-wideres-pgdHE-wide10.pt")
    parser.add_argument('--gpu_id', type=str, default="0")
    parser.add_argument('--awp-warmup', type=int,default=0,
                        help='We could apply AWP after some epochs for accelerating.')
    parser.add_argument('--awp-gamma', type=float, default=0.005,
                        help='whether or not to add parametric noise')
    parser.add_argument('--beta', default=6.0, type=float,
                        help='regularization, i.e., 1/lambda in TRADES')
    return parser.parse_args()


def train_adv_epoch(model, args, train_loader, device, optimizer, epoch,awp_adversary):
    model.train()
    corrects_adv, corrects = 0, 0
    data_num = 0
    loss_sum = 0
    with trange(int(len(train_loader.dataset))) as pbar:
        for batch_idx, (data, target) in enumerate(train_loader):
            x_natural, target = data.to(device), target.to(device)
            data_num += x_natural.shape[0]

            x_adv = perturb_input(model, x_natural, args.step_size, args.epsilon, args.perturb_steps,
                                  distance='l_inf')
            model.train()
            # calculate adversarial weight perturbation
            if epoch >= args.awp_warmup:
                awp = awp_adversary.calc_awp(inputs_adv=x_adv,
                                             inputs_clean=x_natural,
                                             targets=target,
                                             beta=args.beta)
                awp_adversary.perturb(awp)

            optimizer.zero_grad()
            logits_adv = model(x_adv)
            logits_natural = model(x_natural)
            loss_robust = F.kl_div(F.log_softmax(logits_adv, dim=1),
                                   F.softmax(logits_natural, dim=1),
                                   reduction='batchmean')
            # calculate natural loss and backprop

            loss_natural = F.cross_entropy(logits_natural, target)
            loss = loss_natural + args.beta * loss_robust

            # update the parameters at last
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch >= args.awp_warmup:
                awp_adversary.restore(awp)

            loss_sum += loss.item()
            with torch.no_grad():
                model.eval()
                pred_adv = logits_adv.argmax(dim=1)
                pred = torch.argmax(model(x_natural), dim=1)
                corrects_adv += (pred_adv == target).float().sum()
                corrects += (pred == target).float().sum()
            pbar.set_description(f"Train Epoch:{epoch}, Loss:{loss.item():.3f}, " +
                                 f"acc:{corrects / float(data_num):.4f}, " +
                                 f"r_acc:{corrects_adv / float(data_num):.4f}")
            pbar.update(x_natural.shape[0])
    acc, adv_acc = corrects / float(data_num), corrects_adv / float(data_num)
    mean_loss = loss_sum / float(batch_idx + 1)
    return acc, adv_acc, mean_loss


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# 防御任务用的，可以计算出来用PGD对抗方法训练的模型的准确率，鲁棒准确率等
if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    gpu_num = max(len(args.gpu_id.split(',')), 1)

    model_name = 'wide_resnet28'
    log_dir = "logs/%s_%s" % (time.strftime("%b%d-%H%M", time.localtime()), model_name)
    check_mkdir(log_dir)
    log = Logger(log_dir + '/train.log')
    log.print(args)

    device = torch.device('cuda')
    model = WideResNet28().to(device)
    # model.load_state_dict(torch.load('logs/Jun11-1035_defense_model/defense_model_e99_0.8187_0.4883-final.pt'))
    model = nn.DataParallel(model, device_ids=[i for i in range(gpu_num)])

    proxy = WideResNet28().to(device)
    proxy = nn.DataParallel(proxy, device_ids=[i for i in range(gpu_num)])
    proxy_optim = optim.SGD(proxy.parameters(), lr=args.lr)
    awp_adversary = TradesAWP(model=model, proxy=proxy, proxy_optim=proxy_optim, gamma=args.awp_gamma)

    train_loader, test_loader = prepare_cifar(args.batch_size, args.test_batch_size)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr,  weight_decay=args.weight_decay)
    best_epoch, best_robust_acc = 0, 0.
    for e in range(args.epoch):
        adjust_learning_rate(optimizer, e)
        train_acc, train_robust_acc, loss = train_adv_epoch(model, args, train_loader, device, optimizer, e,awp_adversary)
        # if e % 3 == 0 or (74 <= e <= 80):
        test_acc, test_robust_acc, _ = eval_model_pgd(model, test_loader, device, args.step_size, args.epsilon,
                                                          args.perturb_steps*2)
        # else:
        #     test_acc, _ = eval_model(model, test_loader, device)
        if test_robust_acc > best_robust_acc:
            best_robust_acc, best_epoch = test_robust_acc, e
        if e > 50:
            torch.save(model.module.state_dict(),
                       os.path.join(log_dir, f"{model_name}-e{e}-{test_acc:.4f}_{test_robust_acc:.4f}-best.pt"))
        log.print(f"Epoch:{e}, loss:{loss:.5f}, train_acc:{train_acc:.4f}, train_robust_acc:{train_robust_acc:.4f},  " +
                  f"test_acc:{test_acc:.4f}, test_robust_acc:{test_robust_acc:.4f}, " +
                  f"best_robust_acc:{best_robust_acc:.4f} in epoch {best_epoch}.")

    torch.save(model.module.state_dict(),
               f"{log_dir}/{model_name}_e{args.epoch - 1}_{test_acc:.4f}_{test_robust_acc:.4f}-final.pt")
