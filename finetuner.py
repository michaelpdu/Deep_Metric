# coding=utf-8
from __future__ import print_function, absolute_import
import time
from utils import AverageMeter, orth_reg
import torch
from torch.autograd import Variable
from torch.backends import cudnn

cudnn.benchmark = True
use_gpu = torch.cuda.is_available()


def train(epoch, model, criterion, optimizer, train_loader, args):

    losses = AverageMeter()
    batch_time = AverageMeter()
    accuracy = AverageMeter()
    pos_sims = AverageMeter()
    neg_sims = AverageMeter()

    end = time.time()

    freq = min(args.print_freq, len(train_loader))

    for i, data_ in enumerate(train_loader, 0):

        anchors = data_[0]
        positives = data_[1]
        negatives = data_[2]

        # wrap them in Variable
        anchors = Variable(anchors).cuda() if use_gpu else Variable(anchors)
        positives = Variable(positives).cuda() if use_gpu else Variable(positives)
        negatives = Variable(negatives).cuda() if use_gpu else Variable(negatives)

        optimizer.zero_grad()

        anchors_feat = model(anchors)
        positives_feat = model(positives)
        negatives_feat = model(negatives)

        print('')

        # loss, inter_, dist_ap, dist_an = criterion(embed_feat, labels)
        #
        # if args.orth_reg != 0:
        #     loss = orth_reg(net=model, loss=loss, cof=args.orth_reg)
        #
        # loss.backward()
        # optimizer.step()
        #
        # # measure elapsed time
        # batch_time.update(time.time() - end)
        # end = time.time()
        #
        # losses.update(loss.item())
        # accuracy.update(inter_)
        # pos_sims.update(dist_ap)
        # neg_sims.update(dist_an)
        #
        # if (i + 1) % freq == 0 or (i+1) == len(train_loader):
        #     print('Epoch: [{0:03d}][{1}/{2}]\t'
        #           'Time {batch_time.avg:.3f}\t'
        #           'Loss {loss.avg:.4f} \t'
        #           'Accuracy {accuracy.avg:.4f} \t'
        #           'Pos {pos.avg:.4f}\t'
        #           'Neg {neg.avg:.4f} \t'.format
        #           (epoch + 1, i + 1, len(train_loader), batch_time=batch_time,
        #            loss=losses, accuracy=accuracy, pos=pos_sims, neg=neg_sims))
        #
        # if epoch == 0 and i == 0:
        #     print('-- HA-HA-HA-HA-AH-AH-AH-AH --')
