import argparse
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
from src.utils import get_data
from src.model import GCN_CAPS_Model
from src.L2Regularization import Regularization
from src.eval_metrics import *
torch.autograd.set_detect_anomaly(True)
os.environ['CUDA_VISIBLE_DEVICES']='0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def adjust_learning_rate(optimizer, epoch, args): # 调整学习率
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# 设置固定的随机数种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True

# 记录输出
class Logger(object):
    def __init__(self, filename='default.txt', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

# 焦点损失函数
class focal_loss(nn.Module):
    def __init__(self, gamma=2, alpha=0.22, reduction='mean'):
        super(focal_loss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 计算二元交叉熵
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')

        # 计算焦点损失
        focal_loss = self.alpha * (1 - inputs) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            focal_loss = torch.mean(focal_loss)
        elif self.reduction == 'sum':
            focal_loss = torch.sum(focal_loss)

        return focal_loss

def train(train_loader, model, criterion, optimizer, epoch, weight_decay, reg_loss,app, args):
    results = []
    truths = []
    model.train() #启用 batch normalization 和 dropout
    total_loss = 0.0
    total_loss2 = 0.0
    total_batch_size = 0
    train_loader = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False, dynamic_ncols=True)
    # for ind, (batch_X, batch_Y, batch_META) in enumerate(train_loader):
    for ind, (batch_X, batch_Y) in enumerate(train_loader):
        # measure data loading time
        # sample_ind, text, audio, video= batch_X
        # text, audio, video = text.cuda(non_blocking=True), audio.cuda(non_blocking=True), video.cuda(non_blocking=True)
        sample_ind, text, audio, video, prompt = batch_X
        text, audio, video, prompt= text.cuda(non_blocking=True), audio.cuda(non_blocking=True), video.cuda(non_blocking=True), prompt.cuda(non_blocking=True)
        batch_Y = batch_Y.cuda(non_blocking=True)
        eval_attr = batch_Y.squeeze(-1)
        batch_size = text.size(0)
        total_batch_size += batch_size
        # preds= model(text, audio, video, batch_size) # 训练
        preds = model(text, audio, video, prompt, batch_size)  # 训练
        # preds, masked_preds = model(text, audio, video, batch_size)  # 训练
        if args.dataset in ['mosi', 'mosei_senti']:
            preds = preds.reshape(-1)
            eval_attr = eval_attr.reshape(-1)
            raw_loss = criterion(preds, eval_attr)
            if weight_decay > 0:
                raw_loss = raw_loss + reg_loss(model)
            results.append(preds)
            truths.append(eval_attr)
        elif args.dataset == 'iemocap':
            preds = preds.view(-1, 2)
            eval_attr = eval_attr.view(-1)
            raw_loss = criterion(preds, eval_attr)
            if weight_decay > 0:
                raw_loss = raw_loss + reg_loss(model)
            results.append(preds)
            truths.append(eval_attr)
        elif args.dataset in ['MMHS150K', 'MMHS150K_prompt', 'MMHS150K_bert', 'MMHS150K+prompt', 'MMHS150K+flant5']:
            preds = preds.reshape(-1, 2)
            # preds = preds.reshape(-1)
            # masked_preds = masked_preds.reshape(-1, 2)
            # eval_attr = eval_attr.reshape(-1, 2)
            eval_attr = eval_attr.view(-1)
            eval_attr1 = torch.stack([(1 - eval_attr).float(), eval_attr.float()], dim=1)
            # print("preds ", preds)
            # print("eval_attr ",eval_attr)
            raw_loss = criterion(preds, eval_attr) #交叉熵
            # raw_loss2 = criterion(masked_preds, eval_attr)  # 交叉熵
            if weight_decay > 0:
                raw_loss = raw_loss + reg_loss(model)
                # raw_loss2 = raw_loss2 + reg_loss(model)
            results.append(preds)
            truths.append(eval_attr)

        total_loss += raw_loss.item() * batch_size
        combined_loss = raw_loss
        # total_loss2 += raw_loss2.item() * batch_size #对抗训练
        # combined_loss2 = raw_loss2
        optimizer.zero_grad()
        combined_loss.backward()
        # combined_loss2.backward() #对抗训练
        # app.perturb()  # 参数扰动
        # preds_noise = model(text, audio, video, batch_size)  # 扰动后预测
        # loss_noise = F.kl_div(F.log_softmax(preds_noise, dim=-1), F.softmax(preds.clone().detach(), dim=-1), reduction='batchmean')
        # # loss_noise = criterion(preds_noise, eval_attr)
        # loss_noise.backward()
        # app.restore()  # 参数恢复
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip) # 梯度裁剪，防止梯度爆炸
        optimizer.step()
        avg_loss = total_loss / total_batch_size
        train_loader.set_postfix({'Avg Loss': avg_loss})

    avg_loss = total_loss / total_batch_size
    results = torch.cat(results)
    truths = torch.cat(truths)
    return avg_loss, results, truths


def validate(loader, model, criterion2, args):
    model.eval() #评估
    results = []
    truths = []
    total_loss = 0.0
    total_batch_size = 0
    with torch.no_grad():
        # for ind, (batch_X, batch_Y, batch_META) in enumerate(loader):
        for ind, (batch_X, batch_Y) in enumerate(loader):
            # sample_ind, text, audio, video = batch_X
            sample_ind, text, audio, video, prompt = batch_X
            # text, audio, video = text.cuda(non_blocking=True), audio.cuda(non_blocking=True), video.cuda(non_blocking=True)
            text, audio, video, prompt = text.cuda(non_blocking=True), audio.cuda(non_blocking=True), video.cuda(non_blocking=True), prompt.cuda(non_blocking=True)
            batch_Y = batch_Y.cuda(non_blocking=True)
            eval_attr = batch_Y.squeeze(-1)   # if num of labels is 1
            batch_size = text.size(0)
            total_batch_size += batch_size
            # preds, masked_preds = model(text, audio, video, batch_size)
            # preds = model(text, audio, video, batch_size)
            preds = model(text, audio, video, prompt, batch_size)
            if args.dataset in ['mosi', 'mosei_senti']:
                preds = preds.reshape(-1)
                eval_attr = eval_attr.reshape(-1)
                raw_loss = criterion(preds, eval_attr)
                results.append(preds)
                truths.append(eval_attr)
                total_loss += raw_loss.item() * batch_size
            elif args.dataset == 'iemocap':
                preds = preds.view(-1, 2)
                eval_attr = eval_attr.view(-1)
                raw_loss = criterion(preds, eval_attr)
                results.append(preds)
                truths.append(eval_attr)
                total_loss += raw_loss.item() * batch_size

            elif args.dataset in ['MMHS150K', 'MMHS150K_prompt', 'MMHS150K_bert', 'MMHS150K+prompt', 'MMHS150K+flant5']:
                preds = preds.reshape(-1, 2)
                # preds = preds.reshape(-1)
                # masked_preds = masked_preds.reshape(-1, 2)
                # eval_attr = eval_attr.reshape(-1, 2)
                eval_attr = eval_attr.view(-1)
                eval_attr1 = torch.stack([(1 - eval_attr).float(), eval_attr.float()], dim=1)
                # print(preds)
                # print(eval_attr)
                raw_loss = criterion2(preds, eval_attr).cuda()
                # raw_loss2 = criterion2(masked_preds, eval_attr)  # 交叉熵
                results.append(preds)
                truths.append(eval_attr)
                total_loss += raw_loss.item() * batch_size

    avg_loss = total_loss / total_batch_size
    results = torch.cat(results)
    truths = torch.cat(truths)
    return avg_loss, results, truths


if __name__ == "__main__":
    seed = 3407
    setup_seed(seed)
    sys.stdout = Logger('result.txt', sys.stdout)
    sys.stderr = Logger('error.txt', sys.stderr)
    parser = argparse.ArgumentParser(description='PyTorch GCN_CAPS Learner')
    # parser.add_argument('--aligned', action='store_true', default=False, help='consider aligned experiment or not')
    parser.add_argument('--aligned', action='store_true', default=True, help='consider aligned experiment or not')
    # parser.add_argument('--dataset', type=str, default='mosi', help='dataset to use')
    parser.add_argument('--dataset', type=str, default='MMHS150K+prompt', help='dataset to use')
    parser.add_argument('--data-path', type=str, default='MMHS150K', help='path for storing the dataset')
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=64, type=int)  # 32
    parser.add_argument('--lr', '--learning-rate', default=2 * 0.00001, type=float, metavar='LR',
                        help='initial learning rate', dest='lr')  # 0.0001
    parser.add_argument('--MULT_d', default=30, type=int, help='the output dimensionality of MULT is 2*MULT_d')  # 30
    parser.add_argument('--vertex_num', default=20, type=int, help='number of vertexes')  # 20
    parser.add_argument('--dim_capsule', default=256, type=int, help='dimension of capsule')  # 32
    parser.add_argument('--routing', default=3, type=int, help='total routing rounds')  # 3
    parser.add_argument('--weight_decay', default=0.001, type=float, help='L2Regularization')  # 0.001
    parser.add_argument('--dropout', default=0.5, type=float, help='dropout in primary capsule in StoG')  # 0.2
    parser.add_argument('--optimizer', default='Adam', type=str)
    parser.add_argument('--clip', type=float, default=1, help='gradient clip value (default: 1)')
    parser.add_argument('--patience', default=3, type=int, help='patience for learning rate decay')
    parser.add_argument('--layers', default=9, type=int, help='the number of layer used')  # 3
    parser.add_argument('--k_1', default=8, type=int,
                        help='the number of adjacent nodes when constructing the graph')  # 该参数可以用在StoG中，将用自注意生成单模态图邻接矩阵替换成只选K个边构造图
    parser.add_argument('--k_2', default=8, type=int,
                        help='the number of adjacent nodes when constructing the hypergraph')
    args = parser.parse_args()  # 把parser中设置的所有"add_argument"给返回到args子类实例当中 #8

    assert args.dataset in ['mosi', 'mosei_senti', 'iemocap',
                            'MMHS150K', 'MMHS150K_prompt', 'MMHS150K_bert', 'MMHS150K+prompt', 'MMHS150K+flant5'], "supported datasets are mosei_senti, mosi and iemocap"

    hyp_params = args
    hyp_params.MULT_d = args.MULT_d
    hyp_params.vertex_num = args.vertex_num
    hyp_params.dim_capsule = args.dim_capsule
    hyp_params.routing = args.routing
    hyp_params.weight_decay = args.weight_decay
    hyp_params.dropout = hyp_params.dropout
    hyp_params.layers = args.layers
    hyp_params.k_1 = args.k_1
    hyp_params.k_2 = args.k_2
    hyp_params.batch_size = args.batch_size
    current_setting = (hyp_params.MULT_d, hyp_params.vertex_num, hyp_params.dim_capsule, hyp_params.routing,
                       hyp_params.dropout, hyp_params.weight_decay, hyp_params.layers, hyp_params.k_1, hyp_params.k_2,
                       args.optimizer, args.batch_size)

    if args.dataset in['MMHS150K','MMHS150K+prompt', 'MMHS150K+flant5']:
        # criterion = nn.CrossEntropyLoss().cuda()
        samples_count = [105325, 29498]
        # samples_count = [52662, 29498]
        samples_weight = 1 / torch.Tensor(samples_count)
        # criterion = nn.BCELoss(weight=samples_weight).cuda()
        criterion = nn.CrossEntropyLoss(weight=samples_weight).cuda()
        criterion2 = nn.CrossEntropyLoss().cuda()
        # criterion = nn.BCELoss(weight=samples_weight).cuda()
        # criterion2 = nn.BCELoss().cuda()
        t_in = 64
        a_in = 64
        v_in = 64
        label_dim = 2
        T_t, T_a, T_v = 70, 70, 49

    # 定义模型
    model = GCN_CAPS_Model(args, label_dim, t_in, a_in, v_in, T_t, T_a, T_v,
                           hyp_params.MULT_d,
                           hyp_params.vertex_num,
                           hyp_params.dim_capsule,
                           hyp_params.routing,
                           hyp_params.dropout).cuda()

    weight_decay = args.weight_decay
    if weight_decay > 0:
        reg_loss = Regularization(model, weight_decay, p=2).cuda()
    else:
        reg_loss = 0
    # 选取优化器
    if args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.lr)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr)
    # 连续patience次验证集损失没有降低时降低学习率
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=args.patience, factor=0.3, verbose=True)
    train_data = get_data(args, args.dataset, 'train')
    valid_data = get_data(args, args.dataset, 'val')
    test_data = get_data(args, args.dataset, 'test')

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,generator=torch.Generator(device='cuda'))
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    if args.dataset in ['MMHS150K',  'MMHS150K+prompt', 'MMHS150K+flant5']:
        best_valid = 1e9
        # best_valid = 0
        mae_best_acc = 2
        mult_a7_best_acc = -1
        mult_a5_best_acc = -1
        corr_best_acc = 0
        best_acc = 0
        best_pre = 0
        best_rec = 0
        best_f1 = 0
        best_rocauc = 0
        patience_acc = 0

        # 保存每轮的效果用于绘图
        train_loss_list = []
        train_acc_list = []
        test_loss_list = []
        test_acc_list = []
        valid_loss_list = []
        valid_acc_list = []
        Neutral_f1_list = []
        Neutral_acc_list = []
        Happy_f1_list = []
        Happy_acc_list = []
        Sad_f1_list = []
        Sad_acc_list = []
        Angry_f1_list = []
        Angry_acc_list = []
        f1_list = []
        acc_list = []
        rec_list = []
        pre_list = []
        # epoch_list = np.arange(1, args.epochs+1)

    # app = APP(model,
    #           emb_names=['StoG', 'GraphAggregate', 'HyperG', 'HyperGraphConv'])
    app = 0
    # 训练
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        # train for one epoch
        train_loss, train_results, train_truth = train(train_loader, model, criterion, optimizer, epoch,
                                                       weight_decay, reg_loss, app, args)
        # validate for one epoch
        valid_loss, valid_results, valid_truth = validate(valid_loader, model, criterion2, args)
        # test for one epoch
        test_loss, test_results, test_truth = validate(test_loader, model, criterion2, args)
        # 连续patience次验证集损失没有降低时降低学习率
        scheduler.step(valid_loss)

        if args.dataset in ['MMHS150K', 'MMHS150K+prompt', 'MMHS150K+flant5']:
            f1_train, acc_train, pre_train, rec_train, roc_auc_train,_ = eval_MMHS150K(train_results, train_truth)
            f1_test, acc_test, pre_test, rec_test, roc_auc_test,test_predsresult = eval_MMHS150K(test_results, test_truth)
            f1_valid, acc_valid, pre_valid, rec_valid, roc_auc_valid,_ = eval_MMHS150K(valid_results, valid_truth)

        # 记录每一轮的指标用于画图
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        valid_loss_list.append(valid_loss)

        # 显示每一轮的损失值并记录最好的结果
        if args.dataset in ['MMHS150K','MMHS150K_prompt', 'MMHS150K_bert', 'MMHS150K+prompt', 'MMHS150K+flant5'] :
            train_acc_list.append(acc_train)
            test_acc_list.append(acc_test)
            valid_acc_list.append(acc_valid)

            # metrics = {"default": valid_loss,
            #            "acc": acc_test,
            #            "F": f1_test,
            #            "AUC": roc_auc_test}
            # nni.report_intermediate_result(metrics)

            print(
                'Epoch {:2d} Loss| Train Loss{:5.4f} | Valid Loss {:5.4f} | Test Loss {:5.4f} || Acc| Train Acc {:5.4f} | Valid Acc {:5.4f} | Test Acc {:5.4f}'
                .format(epoch, train_loss, valid_loss, test_loss, acc_train, acc_valid, acc_test))
            print('Epoch {:2d} Precision| Pre Train{:5.4f} | Pre Valid {:5.4f} | Pre Test {:5.4f}'
                  .format(epoch, pre_train, pre_valid, pre_test))
            print('Epoch {:2d} Recall| Rec Train{:5.4f} | Rec Valid {:5.4f} | Rec Test {:5.4f}'
                  .format(epoch, rec_train, rec_valid, rec_test))
            print('Epoch {:2d} F1| F1 Train{:5.4f} | F1 Valid {:5.4f} | F1 Test {:5.4f}'
                  .format(epoch, f1_train, f1_valid, f1_test))
            print('Epoch {:2d} ROC-AUC| ROC-AUC Train{:5.4f} | ROC-AUC Valid {:5.4f} | ROC-AUC Test {:5.4f}'
                  .format(epoch, roc_auc_train, roc_auc_valid, roc_auc_test))
            # if acc_valid > best_valid:
            if valid_loss < best_valid:
                if args.aligned:
                    print('aligned {} dataset | acc improved! saving model to aligned_{}_best_model.pkl'
                          .format(args.dataset, args.dataset))
                    torch.save(model, 'aligned_{}_best_model.pkl'.format(args.dataset))
                else:
                    print('unaligned {} dataset | acc improved! saving model to unaligned_{}_best_model.pkl'
                          .format(args.dataset, args.dataset))
                    torch.save(model, 'unaligned_{}_best_model.pkl'.format(args.dataset))
                best_valid = valid_loss
                # best_valid = acc_valid
                best_acc = acc_test
                best_pre = pre_test
                best_rec = rec_test
                best_f1 = f1_test
                best_rocauc = roc_auc_test
                patience_acc = 0
                np.save('test_result.npy', test_predsresult)
            else:
                patience_acc += 1
            if patience_acc > 10:
                epoch_list = np.arange(1, epoch + 2)
                break

    if args.dataset in ['MMHS150K',  'MMHS150K+prompt', 'MMHS150K+flant5']:
        print("hyper-parameters: MULT_d, vertex_num, dim_capsule, routing, dropout, weight_decay, layers, k_1, k_2, "
              "optimizer, batch_size", current_setting)
        print("Best Acc: {:5.4f}".format(best_acc))
        print("precision: {:5.4f}".format(best_pre))
        print("recall: {:5.4f}".format(best_rec))
        print("fscore: {:5.4f}".format(best_f1))
        print("roc-auc: {:5.4f}".format(best_rocauc))

        print('-' * 50)

        # 绘制折线图
        train_loss_list = np.around(train_loss_list, decimals=4)
        train_acc_list = np.around(train_acc_list, decimals=4)
        test_loss_list = np.around(test_loss_list, decimals=4)
        test_acc_list = np.around(test_acc_list, decimals=4)
        valid_loss_list = np.around(valid_loss_list, decimals=4)
        valid_acc_list = np.around(valid_acc_list, decimals=4)

        plt.figure(num=1)
        plt.plot(epoch_list, train_loss_list, "r", marker='D', markersize=1, label="train_loss")
        # plt.plot(epoch_list, test_loss_list, "g", marker='D', markersize=1, label="test_loss")
        # 绘制坐标轴标签
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.title("Loss")
        plt.legend(loc="upper left")
        # 保存图片
        plt.savefig("train_loss.jpg")

        plt.figure(num=2)
        # plt.plot(epoch_list, train_loss_list, "r", marker='D', markersize=1, label="train_loss")
        plt.plot(epoch_list, valid_loss_list, "g", marker='D', markersize=1, label="valid_loss")
        # plt.plot(epoch_list, test_loss_list, "g", marker='D', markersize=1, label="test_loss")
        # 绘制坐标轴标签
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.title("Loss")
        plt.legend(loc="upper left")
        # 保存图片
        plt.savefig("valid_loss.jpg")

        plt.figure(num=3)
        # plt.plot(epoch_list, train_loss_list, "r", marker='D', markersize=1, label="train_loss")
        # plt.plot(epoch_list, valid_loss_list, "r", marker='D', markersize=1, label="valid_loss")
        plt.plot(epoch_list, test_loss_list, "g", marker='D', markersize=1, label="test_loss")
        # 绘制坐标轴标签
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.title("Loss")
        plt.legend(loc="upper left")
        # 保存图片
        plt.savefig("test_loss.jpg")

        plt.figure(num=4)
        # plt.plot(epoch_list, train_acc_list, label="train_acc_2")
        plt.plot(epoch_list, valid_acc_list, label="valid_acc")
        plt.xlabel("epochs")
        plt.ylabel("acc")
        plt.title("Acc")
        plt.legend(loc="upper left")
        plt.savefig("valid_acc.jpg")
        plt.show()

        plt.figure(num=5)
        # plt.plot(epoch_list, train_acc_list, label="train_acc_2")
        plt.plot(epoch_list, test_acc_list, label="test_acc")
        plt.xlabel("epochs")
        plt.ylabel("acc")
        plt.title("Acc")
        plt.legend(loc="upper left")
        plt.savefig("test_acc.jpg")
        plt.show()

    elif args.dataset == 'iemocap':
        train_loss_list = np.around(train_loss_list, decimals=4)
        test_loss_list = np.around(test_loss_list, decimals=4)
        Neutral_f1_list = np.around(Neutral_f1_list, decimals=4)
        Neutral_acc_list = np.around(Neutral_acc_list, decimals=4)
        Happy_f1_list = np.around(Happy_f1_list, decimals=4)
        Happy_acc_list = np.around(Happy_acc_list, decimals=4)
        Sad_f1_list = np.around(Sad_f1_list, decimals=4)
        Sad_acc_list = np.around(Sad_acc_list, decimals=4)
        Angry_f1_list = np.around(Angry_f1_list, decimals=4)
        Angry_acc_list = np.around(Angry_acc_list, decimals=4)

        plt.figure(num=1)
        plt.plot(epoch_list, train_loss_list, "r", marker='D', markersize=1, label="train_loss")
        plt.plot(epoch_list, test_loss_list, "g", marker='D', markersize=1, label="test_loss")
        # 绘制坐标轴标签
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.title("Loss")
        # 调用 text()在图像上绘制注释文本
        # x1、y1表示文本所处坐标位置，ha参数控制水平对齐方式, va控制垂直对齐方式，str(y1)表示要绘制的文本
        for x1, y1 in zip(epoch_list, train_loss_list):
            plt.text(x1, y1, y1, ha='center', va='bottom', fontsize=10)
        for x1, y1 in zip(epoch_list, test_loss_list):
            plt.text(x1, y1, y1, ha='center', va='bottom', fontsize=10)
        plt.legend(loc="upper left")
        # 保存图片
        plt.savefig("loss.jpg")

        plt.figure(num=2)
        plt.plot(epoch_list, Neutral_f1_list, "r", marker='D', markersize=1, label="f1")
        plt.plot(epoch_list, Neutral_acc_list, "g", marker='D', markersize=1, label="acc")
        # 绘制坐标轴标签
        plt.xlabel("epochs")
        plt.ylabel("F1/Acc")
        plt.title("Neutral")
        # 调用 text()在图像上绘制注释文本
        # x1、y1表示文本所处坐标位置，ha参数控制水平对齐方式, va控制垂直对齐方式，str(y1)表示要绘制的文本
        for x1, y1 in zip(epoch_list, Neutral_f1_list):
            plt.text(x1, y1, y1, ha='center', va='bottom', fontsize=10)
        for x1, y1 in zip(epoch_list, Neutral_acc_list):
            plt.text(x1, y1, y1, ha='center', va='bottom', fontsize=10)
        plt.legend(loc="upper left")
        # 保存图片
        plt.savefig("Neutral.jpg")

        plt.figure(num=3)
        plt.plot(epoch_list, Happy_f1_list, "r", marker='D', markersize=1, label="f1")
        plt.plot(epoch_list, Happy_acc_list, "g", marker='D', markersize=1, label="acc")
        # 绘制坐标轴标签
        plt.xlabel("epochs")
        plt.ylabel("F1/Acc")
        plt.title("Happy")
        # 调用 text()在图像上绘制注释文本
        # x1、y1表示文本所处坐标位置，ha参数控制水平对齐方式, va控制垂直对齐方式，str(y1)表示要绘制的文本
        for x1, y1 in zip(epoch_list, Happy_f1_list):
            plt.text(x1, y1, y1, ha='center', va='bottom', fontsize=10)
        for x1, y1 in zip(epoch_list, Happy_acc_list):
            plt.text(x1, y1, y1, ha='center', va='bottom', fontsize=10)
        plt.legend(loc="upper left")
        # 保存图片
        plt.savefig("Happy.jpg")

        plt.figure(num=4)
        plt.plot(epoch_list, Sad_f1_list, "r", marker='D', markersize=1, label="f1")
        plt.plot(epoch_list, Sad_acc_list, "g", marker='D', markersize=1, label="acc")
        # 绘制坐标轴标签
        plt.xlabel("epochs")
        plt.ylabel("F1/Acc")
        plt.title("Sad")
        # 调用 text()在图像上绘制注释文本
        # x1、y1表示文本所处坐标位置，ha参数控制水平对齐方式, va控制垂直对齐方式，str(y1)表示要绘制的文本
        for x1, y1 in zip(epoch_list, Sad_f1_list):
            plt.text(x1, y1, y1, ha='center', va='bottom', fontsize=10)
        for x1, y1 in zip(epoch_list, Sad_acc_list):
            plt.text(x1, y1, y1, ha='center', va='bottom', fontsize=10)
        plt.legend(loc="upper left")
        # 保存图片
        plt.savefig("Sad.jpg")

        plt.figure(num=5)
        plt.plot(epoch_list, Angry_f1_list, "r", marker='D', markersize=1, label="f1")
        plt.plot(epoch_list, Angry_acc_list, "g", marker='D', markersize=1, label="acc")
        # 绘制坐标轴标签
        plt.xlabel("epochs")
        plt.ylabel("F1/Acc")
        plt.title("Angry")
        # 调用 text()在图像上绘制注释文本
        # x1、y1表示文本所处坐标位置，ha参数控制水平对齐方式, va控制垂直对齐方式，str(y1)表示要绘制的文本
        for x1, y1 in zip(epoch_list, Angry_f1_list):
            plt.text(x1, y1, y1, ha='center', va='bottom', fontsize=10)
        for x1, y1 in zip(epoch_list, Angry_acc_list):
            plt.text(x1, y1, y1, ha='center', va='bottom', fontsize=10)
        plt.legend(loc="upper left")
        # 保存图片
        plt.savefig("Angry.jpg")
