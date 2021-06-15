import numpy as np
import torch


class DeepFoolAttack(object):
    def __init__(self,step_size,epsilon,perturb_steps):
        self.step_size=step_size
        self.nb_candidate = 10
        self.overshoot = 0.02
        self.epsilon = epsilon
        self.max_iter = perturb_steps
        self.clip_min = 0.0
        self.clip_max = 1.0

    def __call__(self, model, x, yyy):
        device = x.device

        with torch.no_grad():
            logits = model(x)
        self.nb_classes = logits.size(-1)

        adv_x = x.clone().requires_grad_()

        iteration = 0
        logits = model(adv_x)
        current = logits.argmax(dim=1)
        if current.size() == ():
            current = torch.tensor([current])
        w = torch.squeeze(torch.zeros(x.size()[1:])).to(device)
        r_tot = torch.zeros(x.size()).to(device)
        original = current

        while ((current == original).any and iteration < self.max_iter):
            predictions_val = logits.topk(self.nb_classes)[0]
            gradients = torch.stack(jacobian(predictions_val, adv_x, self.nb_classes), dim=1)
            with torch.no_grad():
                for idx in range(x.size(0)):
                    pert = float('inf')
                    if current[idx] != original[idx]:
                        continue
                    for k in range(1, self.nb_classes):
                        w_k = gradients[idx, k, ...] - gradients[idx, 0, ...]
                        f_k = predictions_val[idx, k] - predictions_val[idx, 0]
                        pert_k = (f_k.abs() + 0.00001) / w_k.view(-1).norm()
                        if pert_k < pert:
                            pert = pert_k
                            w = w_k

                    r_i = pert * w / w.view(-1).norm()
                    r_tot[idx, ...] = r_tot[idx, ...] + r_i

            adv_x = torch.clamp(r_tot + x, self.clip_min, self.clip_max).requires_grad_()
            adv_x = torch.min(torch.max(adv_x, x - 8/255), x + 8/255)
            logits = model(adv_x)
            current = logits.argmax(dim=1)
            if current.size() == ():
                current = torch.tensor([current])
            iteration = iteration + 1
        adv_x = torch.clamp((1 + self.overshoot) * r_tot + x, self.clip_min, self.clip_max)
        adv_x = torch.min(torch.max(adv_x, x - self.epsilon), x + self.epsilon)
        return adv_x


def jacobian(predictions, x, nb_classes):
    list_derivatives = []

    for class_ind in range(nb_classes):
        outputs = predictions[:, class_ind]
        derivatives, = torch.autograd.grad(outputs, x, grad_outputs=torch.ones_like(outputs), retain_graph=True)
        list_derivatives.append(derivatives)

    return list_derivatives
