from models import WideResNet, ResNet18, ResNet34, SmallResNet, WideResNet28, WideResNet34
import torch
import torchvision
import torchvision.transforms as transforms
from ChangedResnet18 import ChangedResnet18


def filter_state_dict(state_dict):
    from collections import OrderedDict

    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'sub_block' in k:
            continue
        if 'module' in k:
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def get_model_for_attack(model_name):
    if model_name == 'model1':
        model = ResNet34()
        model.load_state_dict(torch.load("models/weights/resnet34.pt"))
    elif model_name == 'model2':
        model = ResNet18()
        model.load_state_dict(torch.load('models/weights/resnet18_AT.pt'))
    elif model_name == 'model3':
        model = SmallResNet()
        model.load_state_dict(torch.load('models/weights/res_small.pth'))
    elif model_name == 'model4':
        model = WideResNet34()
        model.load_state_dict(filter_state_dict(torch.load('models/weights/trades_wide_resnet.pt')))
    elif model_name == 'model5':
        model = WideResNet()
        model.load_state_dict(torch.load('models/weights/wideres34-10-pgdHE.pt'))
    elif model_name == 'model6':
        model = WideResNet28()
        model.load_state_dict(filter_state_dict(torch.load('models/weights/RST-AWP_cifar10_linf_wrn28-10.pt')))
    elif model_name == 'Resnet18':
        model = ResNet18()
        model.load_state_dict(torch.load('weight/resnet18-e76-0.8400_0.5001-best.pt'))
    elif model_name == 'ChangedResnet18':
        model = ChangedResnet18()
        model.load_state_dict(torch.load('weight/defense_model-e80-0.8040_0.4946-best.pt'))
    elif model_name == 'WideResnet28':
        model = WideResNet28()
        model.load_state_dict(torch.load('weight/wide_resnet28-e91-0.8640_0.5832-best.pt'))
    return model


if __name__ == '__main__':
    # model = model_dicts['Wu2020Adversarial_extra']
    model = get_model_for_attack('model1')
    # print(model)
    # for name, parameters in model.named_parameters():
    #     print(name, ':', parameters.size())

    # transform_test = transforms.Compose([
    #         transforms.ToTensor(),
    #         ])
    # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
