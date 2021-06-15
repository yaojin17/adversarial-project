import torch
import torch.nn as nn
import torch.nn.functional as F


class PGD20Attack:
    def __init__(self, step_size, epsilon, perturb_steps,
                 random_start=None):
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.random_start = random_start

    def __call__(self, model, x, y):
        model.eval()
        grad_tot = torch.zeros(x.size()).cuda()
        if self.random_start:
            x_adv = x.detach() + self.random_start * torch.randn(x.shape).cuda().detach()+0.008 * torch.randn(x.shape).cuda().detach()
        else:
            x_adv = x.detach()

        for _ in range(self.perturb_steps):
            x_adv.requires_grad_()
            x_adv = x_adv.cuda()
            y = y.cuda()
            grad_tot = grad_tot.cuda()
            with torch.enable_grad():
                loss_c = F.cross_entropy(model(x_adv), y)
                # loss_c = F.mse_loss(model(x_adv), OneHot)
            grad = torch.autograd.grad(loss_c, [x_adv])[0]
            x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x - self.epsilon), x + self.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        return x_adv


def pgd20_attack(model, x, y, step_size, epsilon, perturb_steps,
               random_start=None, distance='l_inf'):
    model.eval()
    batch_size = len(x)
    if random_start:
        x_adv = x.detach() + random_start * torch.randn(x.shape).cuda().detach()
    else:
        x_adv = x.detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_c = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_c, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv
