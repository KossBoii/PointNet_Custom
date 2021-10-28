import argparse
import os
import sys
import logging
import torch
import shutil
import importlib
import augment
from tqdm import tqdm
import numpy as np
import datetime
from pathlib import Path
import time
from data_utils.SDDataset import SDDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

category = ['road', 'car', 'building', 'tree']
category_to_id = {cls: id for id, cls in enumerate(category)}
id_to_category = {id: cls for id, cls in enumerate(category_to_id.keys())}
logger = logging.getLogger("Model")
logger.setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet_sem_seg', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=32, type=int, help='Epoch to run [default: 32]')
    parser.add_argument('--lr', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay rate [default: 1e-4]')
    parser.add_argument('--num_point', type=int, default=4096, help='Point Number [default: 4096]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--ckpt_period', type=float, default=2, help='Checkpoint period')
    parser.add_argument('--test_street', type=int, default=5, help='Which street to use for test, option: 1-6 [default: 5]')
    return parser.parse_args()

def log(str):
    logger.info(str)
    print(str)

def weights_init(model):
    class_name = model.__class__.__name__
    if class_name.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(model.weight.data)
        torch.nn.init.constant_(model.bias.data, 0.0)
    elif class_name.find('Linear') != -1:
        torch.nn.init.xavier_normal_(model.weight.data)
        torch.nn.init.constant_(model.bias.data, 0.0)

def inplace_relu(model):
    classname = model.__class__.__name__
    if classname.find('ReLU') != -1:
        model.inplace=True

def bn_momentum_adjust(model, momentum):
    if isinstance(model, torch.nn.BatchNorm2d) or isinstance(model, torch.nn.BatchNorm1d):
        model.momentum = momentum

def main(args):
    # ---------------------------------------- HYPERPARAMETERS ----------------------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # ---------------------------------------- SETUP DIRECTORY ----------------------------------------
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('sem_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    # ---------------------------------------- SETUP LOGGER ----------------------------------------
    args = parse_args()  
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log('PARAMETER ...')
    log(args)

    # ---------------------------------------- LOADING DATA ----------------------------------------
    root = './data/sight_distance/'
    NUM_CLASSES = 4
    NUM_POINTS = args.num_point
    BATCH_SIZE = args.batch_size

    print("start loading training data ...")
    TRAIN_DATASET = SDDataset(split='train', data_root=root, num_pts=NUM_POINTS, test_street=args.test_street, street_block_size=1.0, sample_rate=1.0, transform=None)
    print("start loading test data ...")
    TEST_DATASET = SDDataset(split='test', data_root=root, num_pts=NUM_POINTS, test_street=args.test_street, street_block_size=1.0, sample_rate=1.0, transform=None)

    print(f'Length: {len(TRAIN_DATASET)}')

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                                                  pin_memory=True, drop_last=True,
                                                  worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                                                 pin_memory=True, drop_last=True)
    weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()

    # ---------------------------------------- SETUP MODEL ----------------------------------------
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/pointnet2_utils.py', str(experiment_dir))

    model = MODEL.get_model(NUM_CLASSES).cuda()
    criterion = MODEL.get_loss().cuda()
    model.apply(inplace_relu)

    # lOADING CHECKPOINTS
    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        log('Use pretrain model')
    except:
        log('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = model.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    best_iou = 0

    # ---------------------------------------- TRAIN MODEL ----------------------------------------
    for epoch in range(start_epoch, args.epoch):
        # TRAIN ON '''Train on chopped scenes'''
        log('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.lr * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        model = model.apply(lambda x: bn_momentum_adjust(x, momentum))
        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        model = model.train()

        for i, (points, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            points[:, :, :3] = augment.rotate_point_cloud_z(points[:, :, :3])
            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            loss = criterion(seg_pred, target, trans_feat, weights)
            loss.backward()
            optimizer.step()

            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINTS)
            loss_sum += loss
        log('Training mean loss: %f' % (loss_sum / num_batches))
        log('Training accuracy: %f' % (total_correct / float(total_seen)))

        # ---------------------------------------- SAVE MODEL EVERY PERIOD ----------------------------------------
        if epoch % args.ckpt_period == 0:
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/model.pth'
            log('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log('Saving model....')

        # ---------------------------------------- EVALUATE MODEL ----------------------------------------
        with torch.no_grad():
            num_batches = len(testDataLoader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            labelweights = np.zeros(NUM_CLASSES)
            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
            classifier = classifier.eval()

            log('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            for i, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                points = points.data.numpy()
                points = torch.Tensor(points)
                points, target = points.float().cuda(), target.long().cuda()
                points = points.transpose(2, 1)

                seg_pred, trans_feat = classifier(points)
                pred_val = seg_pred.contiguous().cpu().data.numpy()
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

                batch_label = target.cpu().data.numpy()
                target = target.view(-1, 1)[:, 0]
                loss = criterion(seg_pred, target, trans_feat, weights)
                loss_sum += loss
                pred_val = np.argmax(pred_val, 2)
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                total_seen += (BATCH_SIZE * NUM_POINTS)
                tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                labelweights += tmp

                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))

            labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6))
            log('eval mean loss: %f' % (loss_sum / float(num_batches)))
            log('eval point avg class IoU: %f' % (mIoU))
            log('eval point accuracy: %f' % (total_correct / float(total_seen)))
            log('eval point avg class acc: %f' % (
                np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))

            iou_per_class_str = '------- IoU --------\n'
            for class_id in range(NUM_CLASSES):
                iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                    id_to_category[class_id] + ' ' * (5 - len(id_to_category[class_id])), labelweights[l - 1],
                    total_correct_class[class_id] / float(total_iou_deno_class[class_id]))

            log(iou_per_class_str)
            log('Eval mean loss: %f' % (loss_sum / num_batches))
            log('Eval accuracy: %f' % (total_correct / float(total_seen)))

            if mIoU >= best_iou:
                best_iou = mIoU
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'class_avg_iou': mIoU,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log('Saving model....')
            log('Best mIoU: %f' % best_iou)
        global_epoch += 1

if __name__ == '__main__':
    print("Hello World")
    args = parse_args()
    main(args)