'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import os
import argparse

from models import *
from utils import progress_bar, adjust_optimizer, get_loss_for_H, update_sram_depth, simplify_BN_parameters, \
                  add_input_hook_for_2nd_conv_layer
import scipy.io as sio


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--data', metavar='DIR', default='/data1/ILSVRC/Data/CLS-LOC/', help='path to dataset')
parser.add_argument('--epochs', default=90, type=int, help='number of total epochs to run')
parser.add_argument('--batch_size', default=256, type=int, help='batch size')
parser.add_argument('--workers', default=4, type=int, help='number of workers')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--resume', type=str, default=None, help='resume from checkpoint')
parser.add_argument('--optimizer', default='SGD', type=str, help='optimizer function used')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float, help='weight decay')
parser.add_argument('--evaluate', type=str, default=None, help='evaluate model FILE on validation set')
parser.add_argument('--save', type=str, help='path to save model')
parser.add_argument('--arch', type=str, default='ResNet_multibit', help='model architecture')
parser.add_argument('--lr_final', type=float, default=-1, help='if positive, exponential lr schedule')
parser.add_argument('--sram-depth', '--sd', default=256, type=int, help='sram depth')
parser.add_argument('--quant-bound', '--qb', default=60, type=int, help='quantization bound')
parser.add_argument('--prob_table', '--pt', default=None, type=str, help='prob table file')
parser.add_argument('--prob_table_9', '--pt9', default=None, type=str, help='prob table file for sum of 9')
parser.add_argument('--noise_std', '--nstd', default=0., type=float, help='amount of noise injected')
parser.add_argument('--noise_9_std', '--n9std', default=0., type=float, help='std of noise injected to ideal quantized output')
parser.add_argument('--noise_9_mean', '--n9mean', default=0., type=float, help='mean of noise injected to ideal quantized output')
parser.add_argument('--runs', default=1, type=int, help='how many runs for stochastic quantization')
parser.add_argument('--save-first-layer-out', '--sf', action='store_true', help='save for HW evaluation')
parser.add_argument('--pact', dest='pact',action='store_true', help='using clipped relu in each stage')
parser.add_argument('--a_lambda', type=float, default=0.001, help='The parameter of alpha L2 regularization')

args = parser.parse_args()

#torch.autograd.set_detect_anomaly(True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
train_loss = 0
train_acc = 0
test_loss = 0
test_acc = 0
start_epoch = args.start_epoch

# Data
num_classes = 1000
traindir = os.path.join(args.data, 'train')
valdir = os.path.join(args.data, 'val_')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

train_sampler = None

trainloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
    num_workers=args.workers, pin_memory=True, sampler=train_sampler)

valloader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=200, shuffle=False,
    num_workers=args.workers, pin_memory=True)

# Model
print('==> Building model..')
model_dict = {
        'ResNet18_a1_w1': ResNet18_quant(1, 1, 1000),
        'ResNet18_a2_w1': ResNet18_quant(1, 2, 1000),
        'ResNet18_a2_w2': ResNet18_quant(2, 2, 1000),
        'ResNet18_a4_w4': ResNet18_quant(4, 4, 1000),
        'ResNet18': ResNet18(1000),
        'ResNet_multibit': resnet18b_quant(num_classes=1000, w_bits=2, a_bits=2)
        }
model = model_dict[args.arch]
if args.evaluate is None:
    print(model)

regime = getattr(model, 'regime', {0: {'optimizer': args.optimizer,
                                     'weight_decay': args.weight_decay,
                                     'momentum': args.momentum,
                                     'lr': args.lr},
                                 150: {'lr': args.lr / 10.},
                                 250: {'lr': args.lr / 100.},
                                 350: {'lr': args.lr / 1000.}})

# update regime if exponential learning rate activated
if not hasattr(model, 'regime'):
    if args.lr_final > 0:
        decay_factor = (args.lr_final / args.lr) ** (1/ (args.epochs - 1.))
        lr = args.lr
        for e in range(1, args.epochs):
            regime[e] = {'lr': lr * decay_factor}
            lr *= decay_factor

model = model.to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

if args.evaluate:
    print(args.batch_size)
    model_path = args.evaluate
    if not os.path.isfile(model_path):
        parser.error('invalid checkpoint: {}'.format(model_path))
    checkpoint = torch.load(model_path)
    if 'net' in checkpoint:
        state_dict_name = 'net'
    elif 'model' in checkpoint:
        state_dict_name = 'model'
    else:
        state_dict_name = 'state_dict'
    model.load_state_dict(checkpoint[state_dict_name], strict=False)
    # acc = checkpoint['acc']
    # print(f'Software basline = {acc}%')
    #print(model)
    args.save = None
elif args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    start_epoch=0
    # best_acc = checkpoint['acc']
    # start_epoch = checkpoint['epoch']
    # for name, v in model.named_parameters():
        # print(v.size())

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr)

# Training
def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        if args.pact:
            reg_alpha = torch.tensor(0.).cuda()
            a_lambda = torch.tensor(args.a_lambda).cuda()
            alpha = []
            for name, param in model.named_parameters():
                if 'alpha' in name and not 'alpha_w' in name:
                    # import pdb;pdb.set_trace()
                    # if param.requires_grad:
                        # param.requires_grad = False
                    alpha.append(param.item())
                    reg_alpha += param.item() ** 2
            loss += a_lambda * (reg_alpha)

        loss.backward()

        for param in model.parameters():
            torch.nn.utils.clip_grad_norm(param, 1.0)

        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.data.copy_(p.org)
        optimizer.step()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.org.copy_(p.data.clamp_(-1,1))
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%%'
            % (train_loss/(batch_idx+1), 100.*correct/total))
    train_acc = 100.*correct/total
    train_loss = train_loss / batch_idx
    return train_loss, train_acc

def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum()
            progress_bar(batch_idx, len(valloader), 'Loss: %.3f | Acc: %.3f%%'
                % (test_loss/(batch_idx+1), 100.*correct/total))
            # if batch_idx == 0:
            #     break

    # Save checkpoint.
    test_acc = 100.*correct/total
    test_loss = test_loss/batch_idx
    acc = 100.*correct/total
    state = {
        'model': model.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if acc > best_acc and args.save:
        torch.save(state, os.path.join('./checkpoint/', args.save))
        best_acc = acc
    if args.save:
        torch.save(state, os.path.join('./checkpoint/', 'model.pth'))
    return test_loss, test_acc

def record_input():
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    Hook = add_input_hook_for_2nd_conv_layer(model)
    Labels = []
    Inputs = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum()
            Labels.append(targets.type(torch.int8))
            Inputs.append(Hook.input[0].type(torch.int8))

            progress_bar(batch_idx, len(valloader), 'Loss: %.3f | Acc: %.3f%%'
                % (test_loss/(batch_idx+1), 100.*correct/total))
    Labels = torch.cat(Labels).cpu().numpy()
    Inputs = torch.cat(Inputs).cpu().numpy()
    sio.savemat('results/CIFAR10.mat', {'Inputs': Inputs, 'Labels': Labels})

update_sram_depth(model, args.sram_depth, args.quant_bound, args.prob_table, args.prob_table_9, args.noise_std, args.noise_9_std, args.noise_9_mean)

if args.evaluate:
    if args.sram_depth > 0:
        simplify_BN_parameters(model)
    if args.save_first_layer_out:
        record_input()
    else:
        if args.runs > 1:
            test_acc = torch.zeros((args.runs,)).cuda()
            for i in range(args.runs):
                test_loss, test_acc[i] = test(0)
            print("Average test accuracy: %.2f (%.2f)" % (test_acc.mean(), test_acc.std()))
        else:
            test_loss, test_acc = test(0)
    exit(0)

for epoch in range(start_epoch, args.epochs):
    optimizer = adjust_optimizer(optimizer, epoch, regime)
    if epoch == start_epoch:
        print(optimizer)
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    print('Epoch: %d/%d | LR: %.4f | Train Loss: %.3f | Train Acc: %.2f | Test Loss: %.3f | Test Acc: %.2f (%.2f)' %
            (epoch+1, args.epochs, optimizer.param_groups[0]['lr'], train_loss, train_acc, test_loss, test_acc, best_acc))
