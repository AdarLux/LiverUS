import argparse
import time
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from utils.profile import count_params
from utils.data_aug import ColorAugmentation
import os
import models
import pandas as pd

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', default='D:/Adar/RunTest', type=str, metavar='DIR', help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='models architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('--config', default='cfgs/local_test.yaml')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate models on validation set')
parser.add_argument('--train_image_list', default='', type=str, help='path to train image list')

parser.add_argument('--input_size', default=224, type=int, help='img crop size')
parser.add_argument('--image_size', default=256, type=int, help='ori img size')

parser.add_argument('--model_name', default='', type=str, help='name of the models')

best_prec1 = 0

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

USE_GPU = torch.cuda.is_available()


class Argu(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.data = 'G:\\Liver US\\Imgs for training'
        self.arch = 'resnet18'
        self.config = 'D:/PycharmProjects/LiverUS/FishNet/cfgs/fishnet150.yaml'
        self.workers = 4
        self.epochs = 100
        self.start_epoch = 0
        self.batch_size = 32
        self.lr = 0.1
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.print_freq = 10
        self.resume = "G:\\Liver US\\Results\\checkpoints\\FishTest_99.pth.tar"
        self.train_image_list = ''
        self.input_size = 224
        self.image_size = 256
        self.model_name = 'FishTest'
        self.evaluate = 1
        self.save_path = 'I:/Liver US/Imgs for training/checkpoints'
        self.orig_script = False

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    for k in topk:
        correct_k = correct[:k].flatten().float().sum(0, keepdim=True)  # AL - flatten() was originally view(-1)
        res = correct_k.mul_(100.0 / batch_size)
    return res


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def validate(val_loader, model, criterion, evaluate=False):
    global time_stp
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        #  pytorch 0.4.0 compatible
        if '0.4.' in torch.__version__:
            with torch.no_grad():
                if USE_GPU:
                    input_var = torch.cuda.FloatTensor(input.cuda())
                    target_var = torch.cuda.LongTensor(target.cuda())
                else:
                    input_var = torch.FloatTensor(input)
                    target_var = torch.LongTensor(target)
        else:  # pytorch 0.3.1 or less compatible
            if USE_GPU:
                input = input.cuda()
                target = target.cuda(non_blocking = True)
            input_var = input
            target_var = target
        # compute output
        # with torch.no_grad():  # AL - disabled this 'with', otherwise you can't call loss.backward()
        output = model(input_var)
        loss = criterion(output, target_var)

        # _, max_res = output.data.topk(1, 1)

        if evaluate:
            if i == 0:
                y_normal = output.data[:, 1]
                y_label = target_var.data
                y_irr = output.data[:, 0]
            else:
                y_normal = torch.cat((y_normal, output.data[:, 1]), 0)
                y_label = torch.cat((y_label, target_var.data), 0)
                y_irr = torch.cat((y_irr, output.data[:, 0]), 0)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target_var)
        losses.update(loss.data, input.size(0))
        top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0 and not evaluate:
            line = 'Test: [{0}/{1}]\t' \
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(i, len(val_loader), batch_time=batch_time,
                                                                   loss=losses, top1=top1)

            with open('I:/Liver US/Imgs for training/logs/{}_{}.log'.format(time_stp, args.arch), 'a+') as flog:
                flog.write('{}\n'.format(line))
                print(line)

    if evaluate:
        np_y_normal = y_normal.t().detach().cpu().numpy()
        np_y_label = y_label.t().detach().cpu().numpy()
        np_y_irr = y_irr.t().detach().cpu().numpy()
        np_y_irr_label = 1 - np_y_label
        d_norm = {'model': np_y_normal, 'label': np_y_label}
        d_irr = {'model': np_y_irr, 'label': np_y_irr_label}
        df_norm = pd.DataFrame(d_norm)
        df_irr = pd.DataFrame(d_irr)
        # df_norm.to_csv('G:/Liver US/Small tumors/NormalClassification_{}_{}.log'.format(time_stp, args.arch),
        #                index=False)
        df_irr.to_csv('G:/Liver US/Small tumors/IrregularClassification_{}_{}.log'.format(time_stp, args.arch),
                       index=False)

    return top1.avg


class Object(object):
    pass


def main():
    global args, best_prec1, USE_GPU
    args = Argu()

    with open(args.config) as f:
        config = yaml.load(f)

    for k, v in config['common'].items():
        setattr(args, k, v)

    # create models
    if args.input_size != 224 or args.image_size != 256:
        image_size = args.image_size
        input_size = args.input_size
    else:
        image_size = 256
        input_size = 224
    print("Input image size: {}, test size: {}".format(image_size, input_size))

    # if "model" in config.keys():
    #     model = models.__dict__[args.arch](**config['model'])
    # else:
    #     model = models.__dict__[args.arch]()
    model = models.__dict__[args.arch](num_cls=2)

    if USE_GPU:
        model = model.cuda()
        model = torch.nn.DataParallel(model)

    count_params(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = 'G:/Liver US/Small tumors'  # os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.orig_script:
        img_size = args.input_size
        ratio = 224.0 / float(img_size)

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                ColorAugmentation(),
                normalize,
            ]))
        val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(int(256 * ratio)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ]))
    else:
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize,
            ]))
        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize,
            ]))


    # if args.distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    #     val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    # else:
    train_sampler = None
    val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=(train_sampler is None), sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    if args.evaluate:
        validate(val_loader, model, criterion, evaluate=True)
        return

    for epoch in range(args.start_epoch, args.epochs):
        # if args.distributed:
        #     train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
        save_name = '{}/{}_{}_best.pth.tar'.format(args.save_path, args.model_name, epoch) if is_best else\
            '{}/{}_{}.pth.tar'.format(args.save_path, args.model_name, epoch)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, filename=save_name)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        #  pytorch 0.4.0 compatible
        if '0.4.' in torch.__version__:
            if USE_GPU:
                input_var = torch.cuda.FloatTensor(input.cuda())
                target_var = torch.cuda.LongTensor(target.cuda())
            else:
                input_var = torch.FloatTensor(input)
                target_var = torch.LongTensor(target)
        else:  # pytorch 0.3.1 or less compatible
            if USE_GPU:
                input = input.cuda()
                target = target.cuda(non_blocking=True)
            input_var = input
            target_var = target

        # compute output
        # with torch.no_grad():  # AL - disabled this 'with', otherwise you can't call loss.backward()
        output = model(input_var)
        loss = criterion(output, target_var)
        prec1 = accuracy(output.data, target_var)

        # measure accuracy and record loss
        reduced_prec1 = prec1.clone()

        top1.update(reduced_prec1[0])

        reduced_loss = loss.data.clone()
        losses.update(reduced_loss)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        #  check whether the network is well connected
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            with open('I:/Liver US/Imgs for training/logs/{}_{}.log'.format(time_stp, args.arch), 'a+') as flog:
                line = 'Epoch: [{0}][{1}/{2}]\t ' \
                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                       'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(epoch, i, len(train_loader),
                        batch_time=batch_time, loss=losses, top1=top1)
                print(line)
                flog.write('{}\n'.format(line))


if __name__ == '__main__':
    time_stp = time.strftime("%Y-%m-%d_%H%M%S", time.localtime())
    main()
