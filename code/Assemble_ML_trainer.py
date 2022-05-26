# code adapted from SPARE3D
# https://github.com/ai4ce/SPARE3D

from __future__ import print_function, division
import gc
import datetime
#import matplotlib.pyplot as plt
import torch
import argparse
import os
import numpy as np
import time
from model import *
from Dataloader import *
import pickle
from tensorboardX import SummaryWriter
parser = argparse.ArgumentParser()
parser.add_argument('--Training_dataroot', default="/scratch/sj3042/data/train", required=False,
                    help='path to training dataset')
parser.add_argument('--Validating_dataroot', default="/scratch/sj3042/data/eval", required=False,
                    help='path to validating dataset')
# D:\dataset\eval
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--niter', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.00002, help='learning rate, default=0.00002')
parser.add_argument('--device', default='cuda:0', help='device')
parser.add_argument('--model_type', default='vgg16', help='|vgg16| |resnet50| |Bagnet33|')
parser.add_argument('--outf', default="/scratch/sj3042/data/assemble_ML_out", help='folder to output log')
parser.add_argument('--ckpt_freq', default=10, help='save model ckpt every k epochs')

opt = parser.parse_args()

device = opt.device

device = opt.device

task_1_hard_model = ThreeV2I_ML(opt.model_type).to(opt.device)

gc.collect()

torch.cuda.empty_cache()


def train_model():
    epoch_loss = 0
    epoch_acc = 0
    batch_loss = 0
    batch_acc = 0
    data_transforms = False
    path = opt.Training_dataroot
    train_data = ThreeV2I_ML_data(path)
    data_train = torch.utils.data.DataLoader(train_data, batch_size=opt.batchSize, shuffle=True)
    batch_loss_list = []
    task_1_hard_model.train()
    for i, (View, input_1, input_2, input_3, input_4, Label) in enumerate(data_train):

        if i % 50 == 0:
            print('-' * 30)
            print("batch: %d" % i)

        optimizer.zero_grad()
        View, input_1, input_2, input_3, input_4, Label = View.to(device), input_1.to(device), input_2.to(
            device), input_3.to(device), input_4.to(device), Label.to(device)

        y_1 = task_1_hard_model(View.float(), input_1.float())
        y_2 = task_1_hard_model(View.float(), input_2.float())
        y_3 = task_1_hard_model(View.float(), input_3.float())
        y_4 = task_1_hard_model(View.float(), input_4.float())
        Y = torch.cat((y_1, y_2, y_3, y_4), axis=1)
        total_loss = torch.zeros(1).to(device)
        for i in range(input_1.shape[0]):
            Index = list(range(4))
            index = Label[i].item()
            Index.remove(index)

            loss_1 = criterion(Y[i][index].reshape(1), Y[i][Index[0]].reshape(1), torch.tensor([-1]).to(device))
            loss_2 = criterion(Y[i][index].reshape(1), Y[i][Index[1]].reshape(1), torch.tensor([-1]).to(device))
            loss_3 = criterion(Y[i][index].reshape(1), Y[i][Index[2]].reshape(1), torch.tensor([-1]).to(device))
            total_loss += (loss_1 + loss_2 + loss_3) / 3 / input_1.shape[0]
        batch_loss += total_loss.item() * input_1.shape[0]
        batch_loss_list.append(total_loss.item() * input_1.shape[0] / len(train_data))
        total_loss.backward()
        optimizer.step()
        batch_acc += (Y.argmin(1) == Label.reshape(input_1.shape[0])).sum().item()
        # manually free the memory
        del y_1, y_2, y_3, y_4, Y, total_loss, input_1, input_2, input_3, input_4, Label
        gc.collect()
        torch.cuda.empty_cache()
    epoch_loss = batch_loss / len(train_data)
    epoch_acc = batch_acc / len(train_data)

    return epoch_loss, epoch_acc, np.array(batch_loss_list)


def Eval():
    eval_loss = 0
    eval_acc = 0
    epoch_eval_loss = 0
    epoch_eval_acc = 0

    data_transforms = False
    path = opt.Validating_dataroot
    eval_data = ThreeV2I_ML_data(path)
    data_eval = torch.utils.data.DataLoader(eval_data, batch_size=opt.batchSize, shuffle=True)
    with torch.no_grad():
        task_1_hard_model.eval()
        for i, (View, input_1, input_2, input_3, input_4, Label) in enumerate(data_eval):
            View, input_1, input_2, input_3, input_4, Label = View.to(device), input_1.to(device), input_2.to(
                device), input_3.to(device), input_4.to(device), Label.to(device)

            y_1 = task_1_hard_model(View.float(), input_1.float())
            y_2 = task_1_hard_model(View.float(), input_2.float())
            y_3 = task_1_hard_model(View.float(), input_3.float())
            y_4 = task_1_hard_model(View.float(), input_4.float())
            Y = torch.cat((y_1, y_2, y_3, y_4), axis=1)
            total_loss = torch.zeros(1).to(device)
            for i in range(input_1.shape[0]):
                Index = list(range(4))
                index = Label[i].item()
                Index.remove(index)

                loss_1 = criterion(Y[i][index].reshape(1), Y[i][Index[0]].reshape(1), torch.tensor([-1]).to(device))
                loss_2 = criterion(Y[i][index].reshape(1), Y[i][Index[1]].reshape(1), torch.tensor([-1]).to(device))
                loss_3 = criterion(Y[i][index].reshape(1), Y[i][Index[2]].reshape(1), torch.tensor([-1]).to(device))
                total_loss += (loss_1 + loss_2 + loss_3) / 3 / input_1.shape[0]
            eval_loss += total_loss.item() * input_1.shape[0]
            eval_acc += (Y.argmin(1) == Label.reshape(input_1.shape[0])).sum().item()
        epoch_eval_loss = eval_loss / len(eval_data)
        epoch_eval_acc = eval_acc / len(eval_data)
    return epoch_eval_loss, epoch_eval_acc


N_EPOCHS = opt.niter
criterion = torch.nn.MarginRankingLoss(margin=2.0).to(device)
optimizer = torch.optim.Adam(task_1_hard_model.parameters(), lr=opt.lr)

batch_loss_history = []
train_acc_list = []
val_acc_list = []
train_loss_list = []
val_loss_list = []

dt = datetime.datetime.now()
dt = dt.strftime('%m_%d_%Y_%I_%M_%S_%p')

log_path = os.path.join(opt.outf, 'ML', opt.model_type, dt)
os.makedirs(log_path, exist_ok=True)

train_params_fp = os.path.join(log_path, 'params.json')
with open(train_params_fp, 'w') as ff:
    json.dump(vars(opt), ff, indent=4)

file = open(log_path + "/" + opt.model_type + "_Lr_" + str(opt.lr) + ".txt", "w")
writer=SummaryWriter(opt.outf)
for epoch in range(N_EPOCHS):
    print('Epoch: %d' % (epoch + 1))
    start_time = time.time()
    train_loss, train_acc, batch_list = train_model()
    valid_loss, valid_acc = Eval()
    writer.add_scalar('train_loss', train_loss, global_step=epoch)
    writer.add_scalar('train_acc', train_acc, global_step=epoch)
    writer.add_scalar('test_loss', valid_loss, global_step=epoch)
    writer.add_scalar('test_acc', valid_acc, global_step=epoch)
    train_acc_list.append(train_acc)
    val_acc_list.append(valid_acc)
    train_loss_list.append(train_loss)
    val_loss_list.append(valid_loss)

    batch_loss_history.append(batch_list)

    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60
    file.write('Epoch: %d' % (epoch + 1))
    file.write(" | time in %d minutes, %d seconds\n" % (mins, secs))
    file.write(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)\n')
    file.write(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)\n')
    file.write("\n")
    file.flush()

    print(" | time in %d minutes, %d seconds\n" % (mins, secs))
    print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)\n')
    print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)\n')
    print("\n")

    if (epoch + 1) % opt.ckpt_freq == 0 and (epoch + 1) != N_EPOCHS:
        torch.save(task_1_hard_model.state_dict(),
                   log_path + "/" + opt.model_type + "_e%d_" % (epoch + 1) + "_Lr_" + str(opt.lr) + ".pth")
writer.close()
file.close()
batch_loss_history = np.array(batch_loss_history)
batch_loss_history = np.concatenate(batch_loss_history, axis=0)

batch_loss_history = batch_loss_history.reshape(len(batch_loss_history))
np.save(log_path + "/" + opt.model_type + "_Lr_" + str(opt.lr) + ".npy", batch_loss_history)
torch.save(task_1_hard_model.state_dict(),
           log_path + "/" + opt.model_type + "_e%d_" % (epoch + 1) + "_Lr_" + str(opt.lr) + ".pth")

#fig, axs = plt.subplots(2, constrained_layout=True)
axs[0].set_title("Train and val acc")
axs[0].plot(train_acc_list, 'tab:blue', label='train')
axs[0].plot(val_acc_list, 'tab:orange', label='val')
axs[0].set_xlabel("iterations")
axs[0].set_ylabel("accuracy")
axs[0].legend()

axs[1].set_title("Train and val loss")
axs[1].plot(train_loss_list, 'tab:blue', linestyle='-.', label='train')
axs[1].plot(val_loss_list, 'tab:orange', linestyle='-.', label='val')
axs[1].set_xlabel("iterations")
axs[1].set_ylabel("loss")
axs[1].legend()

fig.savefig(os.path.join(log_path, "loss_acc.png"))

fp = os.path.join(log_path, "loss_acc.pkl")
loss_acc = [train_acc_list, val_acc_list, train_loss_list, val_loss_list]
with open(fp, 'wb') as f:
    pickle.dump(loss_acc, f)
