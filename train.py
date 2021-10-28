import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from model.pointnetv2 import *
import torch.backends.cudnn as cudnn

def get_parser():
    parser = argparse.ArgumentParser('PointNet Parser')
    parser.add_argument('--batch_size', type=int, help='batch size', default=8)
    parser.add_argument('--epochs', type=int, help='number of epochs', default=10)
    parser.add_argument('--log_dir', type=str, help='directory to save checkpoints and summaries', default='')
    parser.add_argument('--ckpt_path', type=str, help='path to a specific checkpoint to load', default='')
    parser.add_argument('--num_workers', type=int, help='number of workers', default=4)

    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    print('Command Line Arguments: ', args)

    # setup dataset
    train_dataset = MyCustomDataset(train=True)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size= args.batch_size, shuffle=True, num_workers=args.num_workers)

    classifier = PointNet2SemSeg()
    optimizer = optim.Adam(classifier.parameters())







    # training starts here
    print("Train examples: {}".format(train_examples))
    print("Evaluation examples: {}".format(test_examples))
    print("Start training...")
    cudnn.benchmark = True
    classifier.cuda()
    for epoch in range(num_epochs):
        print("--------Epoch {}--------".format(epoch))

        # train one epoch
        classifier.train()
        total_train_loss = 0
        correct_examples = 0
        for batch_idx, data in enumerate(train_dataloader, 0):
            pointcloud, label = data
            pointcloud = pointcloud.permute(0, 2, 1)
            pointcloud, label = pointcloud.cuda(), label.cuda()

            optimizer.zero_grad()
            pred = classifier(pointcloud)

            loss = F.nll_loss(pred, label.view(-1))
            pred_choice = pred.max(1)[1]

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            correct_examples += pred_choice.eq(label.view(-1)).sum().item()
            
        print("Train loss: {:.4f}, train accuracy: {:.2f}%".format(total_train_loss / train_batches, correct_examples / train_examples * 100.0))

        # eval one epoch
        classifier.eval()
        correct_examples = 0
        for batch_idx, data in enumerate(test_dataloader, 0):
            pointcloud, label = data
            pointcloud = pointcloud.permute(0, 2, 1)
            pointcloud, label = pointcloud.cuda(), label.cuda()

            pred = classifier(pointcloud)
            pred_choice = pred.max(1)[1]
            correct = pred_choice.eq(label.view(-1)).sum()
            correct_examples += correct.item()

        print("Eval accuracy: {:.2f}%".format(correct_examples / test_examples * 100.0))