import torch
import torch.nn as nn
import torch.nn.functional as F


# 返回每一行元素第二大元素的下标
def second_max(y_hat):
    _, max_idx = torch.max(y_hat, 1)
    min_emt, _ = torch.min(y_hat, 1)
    for i in range(len(y_hat)):
        y_hat[i][max_idx[i]] = min_emt[i]
    _, second_max_idx = torch.max(y_hat, 1)
    return second_max_idx


class EarlyEndPGDAttack:
    def __init__(self, step_size, epsilon, perturb_steps,
                 random_start=None):
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.random_start = random_start

    def __call__(self, model, x, y):
        model.eval()
        batch_size = len(x)
        grad_tot = torch.zeros(x.size()).cuda()
        if self.random_start:
            x_adv = x.detach() + self.random_start * torch.randn(x.shape).cuda().detach()
        else:
            x_adv = x.detach()
        result = torch.zeros(x.size())
        equal_x = torch.zeros(y.size())
        notsucceed_attack = []
        for i in range(len(y)):
            notsucceed_attack.append(i)

        for n in range(self.perturb_steps):
            x_adv.requires_grad_()
            _, rt = torch.max(model(x_adv), 1)
            equal_x = (rt == y)
            tep = []
            temp1 = []
            temp2 = []
            temp3 = []
            temp4 = []
            for i in range(len(equal_x)):
                if equal_x[i] == 0:
                    result[notsucceed_attack[i]] = x_adv[i]
                else:
                    tep.append(notsucceed_attack[i])
                    temp1.append(x_adv[i])
                    temp2.append(y[i])
                    temp3.append(grad_tot[i])
                    temp4.append(x[notsucceed_attack[i]])
            notsucceed_attack = tep
            if len(temp2) == 0:
                break
            x_adv = torch.zeros([len(temp1), 3, 32, 32])
            grad_tot = torch.zeros([len(temp1), 3, 32, 32])
            x_old = torch.zeros([len(temp1), 3, 32, 32])
            for i in range(len(temp1)):
                x_adv[i] = temp1[i]
                grad_tot[i] = temp3[i]
                x_old[i] = temp4[i]
            y = torch.tensor(temp2)
            x_adv = x_adv.cuda()
            y = y.cuda()
            grad_tot = grad_tot.cuda()
            x_old = x_old.cuda()
            y_hat = model(x_adv).cuda()

            miny, _ = torch.min(y_hat, 1)

            with torch.enable_grad():
                # loss_c = F.cross_entropy(y_hat, y) - F.cross_entropy(y_hat, second_max(y_hat))
                # loss_c = - F.cross_entropy(y_hat, second_max(y_hat))
                loss_c = F.cross_entropy(y_hat, y)
            grad = torch.autograd.grad(loss_c, [x_adv])[0]
            x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach()) + 0.008 * torch.randn(
                x_adv.shape).cuda().detach()
            x_adv = x_adv.detach() + 0.017 * torch.sign(grad_tot.cuda().detach())
            grad_tot += grad
            x_adv = torch.min(torch.max(x_adv, x_old - self.epsilon), x_old + self.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        for i in range(len(notsucceed_attack)):
            result[notsucceed_attack[i]] = x_adv[i]

        return result


