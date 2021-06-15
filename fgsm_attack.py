import torch
import torch.nn as nn
import torch.nn.functional as F


def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0.0, 1.0)
    return perturbed_image


class FGSMAttack:
    def __init__(self, step_size,epsilon,perturb_steps):
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps

    def __call__(self, model, x, y):
        model.eval()

        x.requires_grad_()
        with torch.enable_grad():
            loss_c = F.cross_entropy(model(x), y)
        grad = torch.autograd.grad(loss_c, [x])[0]
        x_adv = fgsm_attack(x, self.epsilon, grad)

        return x_adv
