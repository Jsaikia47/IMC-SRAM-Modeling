'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar, adjust_optimizer, get_loss_for_H, update_sram_depth, simplify_BN_parameters, \
                  add_input_hook_for_2nd_conv_layer
import scipy.io as sio


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--epochs', default=350, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--resume', type=str, default=None, help='resume from checkpoint')
parser.add_argument('--optimizer', default='SGD', type=str, help='optimizer function used')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='weight decay')
parser.add_argument('--evaluate', type=str, default=None, help='evaluate model FILE on validation set')
parser.add_argument('--save', type=str, help='path to save model')
parser.add_argument('--arch', type=str, default='VGGTBN_a1_w1', help='model architecture')
parser.add_argument('--lr_final', type=float, default=-1, help='if positive, exponential lr schedule')
parser.add_argument('--train_batch_size', '--trbs', default=100, type=int, help='train batch size')
parser.add_argument('--test_batch_size', '--tsbs', default=100, type=int, help='test batch size')
parser.add_argument('--sram-depth', '--sd', default=256, type=int, help='sram depth')
parser.add_argument('--quant-bound', '--qb', default=60, type=int, help='quantization bound')
parser.add_argument('--prob_table', '--pt', default=None, type=str, help='prob table file')
parser.add_argument('--prob_table_9', '--pt9', default=None, type=str, help='prob table file for sum of 9')
parser.add_argument('--noise_std', '--nstd', default=0., type=float, help='amount of noise injected to ideal output')
parser.add_argument('--noise_9_std', '--n9std', default=0., type=float, help='std of noise injected to ideal quantized output')
parser.add_argument('--noise_9_mean', '--n9mean', default=0., type=float, help='mean of noise injected to ideal quantized output')
parser.add_argument('--runs', default=1, type=int, help='how many runs for stochastic quantization')
parser.add_argument('--save-first-layer-out', '--sf', action='store_true', help='save for HW evaluation')
parser.add_argument('--simplify_BN', action='store_true', help='simplify BN parameters for HW evaluation')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
train_loss = 0
train_acc = 0
test_loss = 0
test_acc = 0
start_epoch = args.start_epoch

# Data
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=1, pin_memory=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True)

# Model
print('==> Building model..')
model_dict = {
        'VGG_a1_w1': VGG_quant('VGG', 1, 1),
        'VGG_a2_w1': VGG_quant('VGG', 2, 1),
        'VGG_a2_w2': VGG_quant('VGG', 2, 2),
        'VGGT_a1_w1': VGG_quant('VGGT', 1, 1, 256),
        'VGGT_a2_w2': VGG_quant('VGGT', 2, 2, 256),
        'VGGT_a2_w1': VGG_quant('VGGT', 2, 1, 256),
        'VGGBN_a1_w1': VGG_bn_quant('VGG', 1, 1),
        'VGGBN_a2_w1': VGG_bn_quant('VGG', 2, 1),
        'VGGBN_a2_w2': VGG_bn_quant('VGG', 2, 2),
        'VGGTBN_a1_w1': VGG_bn_quant('VGGT', 1, 1, 256),
        'VGGTBN_a2_w2': VGG_bn_quant('VGGT', 2, 2, 256),
        'VGGTBN_a2_w1': VGG_bn_quant('VGGT', 2, 1, 256),
        'VGGABN_a1_w1': VGG_bn_quant('VGGA', 1, 1, 256),
        'VGGABN_a2_w2': VGG_bn_quant('VGGA', 2, 2, 256),
        'VGGABN_a2_w1': VGG_bn_quant('VGGA', 2, 1, 256),
        'VGG': VGG('VGG'),
        'VGG16': VGG('VGG16'),
        'ResNet18_a1_w1': ResNet18_quant(1, 1),
        'ResNet18_a2_w1': ResNet18_quant(1, 2),
        'ResNet18_a2_w2': ResNet18_quant(2, 2),
        'ResNet18_a4_w4': ResNet18_quant(4, 4),
        'ResNet18': ResNet18()
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
    #print(model)
    args.save = None
elif args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['model'], strict=False)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

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
        loss.backward()
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
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%%'
                % (test_loss/(batch_idx+1), 100.*correct/total))

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
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum()
            Labels.append(targets.type(torch.int8))
            Inputs.append(Hook.input[0].type(torch.int8))

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%%'
                % (test_loss/(batch_idx+1), 100.*correct/total))
            # if batch_idx == 0:
            #     break
    Labels = torch.cat(Labels).cpu().numpy()
    Inputs = torch.cat(Inputs).cpu().numpy()
    sio.savemat('results/CIFAR10.mat', {'Inputs': Inputs, 'Labels': Labels})

update_sram_depth(model, args.sram_depth, args.quant_bound, args.prob_table, args.prob_table_9, args.noise_std, args.noise_9_std, args.noise_9_mean)

if args.evaluate:
    if args.sram_depth > 0 and args.simplify_BN:
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
