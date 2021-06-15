# Attack and Defense 
Attack and Defense Code for adversarial training project for Machine Learning in Action course in SJTU 


### Set up:
** Requirement ** `python3`, `pytorch > 1.2`, `tqdm`  
1. Install python  
Recomand install anaconda, see https://www.anaconda.com/products/individual#Downloads
2. Create new environment and install pytorch, tqdm:
```
conda create -n t17 python=3.8
conda install pytorch=1.7 torchvision cudatoolkit=10.2 -c pytorch
pip install tqdm
```
### Notes
1. About args  
代码中使用的是Python包`argparse`，来在命令行解析参数。例如设置attack的model: `python attack_main.py --model_name=model1`  
2. About gpu  
推荐使用GPU训练，多卡训练，需设置gpu_id, eg: `python pgd_train.py --gpu_id=0,1`.
在这里使用 Pytorch 的nn.DataParallel 实现多卡并行， 涉及代码如下。nn.DataParallel 实际上 wrap the pytorch model as `model.module`.
```
device = torch.device('cuda')
model = ResNet18().to(device)    
model = nn.DataParallel(model, device_ids=[i for i in range(gpu_num)])
```
3. About log  
关于日志保存，请参见代码中`Logger` class, 也可自行实现日志部分  


# Attack Task
In this task, you need to run attack algorithm to attack provided 6 models.  
Note that we use constraint of `l-inf` norm distance `< 8./255`. 

### Dataset: CIFAR10
Use `prepare_cifar` in utils.py to get `train_loader` and `test_loader` of CIFAR10.  

```python
from utils import prepare_cifar
train_loader, test_loader = prepare_cifar(batch_size = 128, test_batch_size = 256)
```

### Defense models
1. model1:  vanilla resnet34
2. model2:  PGD adversarial trained resnet18
3. model3:  Unkonown resnet
4. model4:  [TRADES](https://arxiv.org/abs/1901.08573)
5. model5:  [PGD_HE](https://arxiv.org/abs/2002.08619)
6. model6:  [RST_AWP](https://arxiv.org/abs/2004.05884)

### 运行我们的模型
你可以·像pgd_attack.py 里的PGDAttack class一样调用一个攻击类。 类似： 
```python
attack = FWEEPGDAttack(args.step_size, args.epsilon, args.perturb_steps)

```
然后可以通过这样调用攻击算法,生成对抗样本：
```python
x_adv = attack(model, x, label)
```
最后，你可以使用 `eval_model_with_attack` in eval_model.py 来测试你调用的攻击算法。 
```python
natural_acc, robust_acc, distance = eval_model_with_attack(model, test_loader, attack, device)
```

### Test your attack 
Run attack_main.py to test your attack, set model_name to [model1, model2, model3, model4, model5, model6]. 
And set method_name to ['FWEEPGDAttack','EarlyEndPGDAttack','FGSMAttack','DeepFoolAttack','SemiTargetedPGDAttack','BlackBoxAttack','PGD20Attack'].
Like:
```sh
python attack_main.py --model_name=model1 --method_name='FWEEPGDAttack'
```



# Defense Task
In this task, we train a robust model under l-inf attack(8/255) on CIFAR10.  

### Evaluate your model
You can use various attack methods to evaluate the robustness of the model.   
Include PGD attack, and others.  

### How to run PGD attack to test robustness of the model
1. Open attack_main.py, specify how to load the model.  
  A example in attack_main.py:
```python
model =  WideResNet().to(device)  
model = nn.DataParallel(model, device_ids=[i for i in range(gpu_num)])
model.load_state_dict(torch.load(args.model_path)['state_dict'], strict=False)
```
2. Run attack_main.py:  
```python
python attack_main.py --model_path=your_weight_path --gpu_id=0      # eg. For multiple gpus, set --gpu_id=1,2,3
```
It will give natural acc and robust acc for the model.

### A Defense Baseline: Adversarial training   

```
train.py   # python train.py 可直接复现我们的模型训练过程
infer.py   # python inter.py  可直接测试 cifar10 test acc，给出natural acc.
attack.py  # python attack.py 运行攻击我们的模型， 请参考attack_main.py 把被攻击模型换成你的。
our_model_name.pth  #我们训练的模型weight
```
train.py,infer.py,attack.py中默认模型均为Trades-AWP方法训练得到的WideResnet28模型，可直接运行这三个文件来进行测试，attack.py默认攻击方法为pgd20，可参考上面的方法按需设置攻击方法。

如果想要测试其他两个防御模型，建议使用attack_main.py进行测试，注意设置model_name为'Resnet18'或者'ChangedResnet18'，其中'Resnet18'为更改训练参数所得的模型，'ChangedResnet18'为尝试使用梯度掩蔽训练得到的模型，攻击方法可按需求参考上面的方式设置。

