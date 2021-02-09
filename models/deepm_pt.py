import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data

import time, json, datetime
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


class DeepFM(nn.Module):
    def __init__(self, cate_fea_nuniqs, nume_fea_size=0, emb_size=8,
                 hid_dims=[256, 128], num_classes=1, dropout=[0.2, 0.2]):
        """
        cate_fea_nuniqs: 类别特征的唯一值个数列表，也就是每个类别特征的vocab_size所组成的列表
        nume_fea_size: 数值特征的个数，该模型会考虑到输入全为类别型，即没有数值特征的情况
        """
        super().__init__()
        self.cate_fea_size = len(cate_fea_nuniqs)
        self.nume_fea_size = nume_fea_size

        """FM部分"""
        # 一阶
        if self.nume_fea_size != 0:
            self.fm_1st_order_dense = nn.Linear(self.nume_fea_size, 1)  # 数值特征的一阶表示
        self.fm_1st_order_sparse_emb = nn.ModuleList([
            nn.Embedding(voc_size, 1) for voc_size in cate_fea_nuniqs])  # 类别特征的一阶表示

        # 二阶
        self.fm_2nd_order_sparse_emb = nn.ModuleList([
            nn.Embedding(voc_size, emb_size) for voc_size in cate_fea_nuniqs])  # 类别特征的二阶表示

        """DNN部分"""
        self.all_dims = [self.cate_fea_size * emb_size] + hid_dims
        self.dense_linear = nn.Linear(self.nume_fea_size, self.cate_fea_size * emb_size)  # 数值特征的维度变换到FM输出维度一致
        self.relu = nn.ReLU()
        # for DNN
        for i in range(1, len(self.all_dims)):
            setattr(self, 'linear_' + str(i), nn.Linear(self.all_dims[i - 1], self.all_dims[i]))
            setattr(self, 'batchNorm_' + str(i), nn.BatchNorm1d(self.all_dims[i]))
            setattr(self, 'activation_' + str(i), nn.ReLU())
            setattr(self, 'dropout_' + str(i), nn.Dropout(dropout[i - 1]))
        # for output
        self.dnn_linear = nn.Linear(hid_dims[-1], num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X_sparse, X_dense=None):
        """
        X_sparse: 类别型特征输入  [bs, cate_fea_size]
        X_dense: 数值型特征输入（可能没有）  [bs, dense_fea_size]
        """

        """FM 一阶部分"""
        fm_1st_sparse_res = [emb(X_sparse[:, i].unsqueeze(1)).view(-1, 1)
                             for i, emb in enumerate(self.fm_1st_order_sparse_emb)]
        fm_1st_sparse_res = torch.cat(fm_1st_sparse_res, dim=1)  # [bs, cate_fea_size]
        fm_1st_sparse_res = torch.sum(fm_1st_sparse_res, 1, keepdim=True)  # [bs, 1]

        if X_dense is not None:
            fm_1st_dense_res = self.fm_1st_order_dense(X_dense)
            fm_1st_part = fm_1st_sparse_res + fm_1st_dense_res
        else:
            fm_1st_part = fm_1st_sparse_res  # [bs, 1]

        """FM 二阶部分"""
        fm_2nd_order_res = [emb(X_sparse[:, i].unsqueeze(1)) for i, emb in enumerate(self.fm_2nd_order_sparse_emb)]
        fm_2nd_concat_1d = torch.cat(fm_2nd_order_res, dim=1)  # [bs, n, emb_size]  n为类别型特征个数(cate_fea_size)

        # 先求和再平方
        sum_embed = torch.sum(fm_2nd_concat_1d, 1)  # [bs, emb_size]
        square_sum_embed = sum_embed * sum_embed  # [bs, emb_size]
        # 先平方再求和
        square_embed = fm_2nd_concat_1d * fm_2nd_concat_1d  # [bs, n, emb_size]
        sum_square_embed = torch.sum(square_embed, 1)  # [bs, emb_size]
        # 相减除以2
        sub = square_sum_embed - sum_square_embed
        sub = sub * 0.5  # [bs, emb_size]

        fm_2nd_part = torch.sum(sub, 1, keepdim=True)  # [bs, 1]

        """DNN部分"""
        dnn_out = torch.flatten(fm_2nd_concat_1d, 1)  # [bs, n * emb_size]

        if X_dense is not None:
            dense_out = self.relu(self.dense_linear(X_dense))  # [bs, n * emb_size]
            dnn_out = dnn_out + dense_out  # [bs, n * emb_size]

        for i in range(1, len(self.all_dims)):
            dnn_out = getattr(self, 'linear_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'batchNorm_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'activation_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'dropout_' + str(i))(dnn_out)

        dnn_out = self.dnn_linear(dnn_out)  # [bs, 1]
        out = fm_1st_part + fm_2nd_part + dnn_out  # [bs, 1]
        out = self.sigmoid(out)
        return out


if __name__ == '__main__':
    data = pd.read_csv("~/Downloads/criteo_sample_50w.csv")

    dense_features = [f for f in data.columns.tolist() if f[0] == "I"]
    sparse_features = [f for f in data.columns.tolist() if f[0] == "C"]

    data[sparse_features] = data[sparse_features].fillna('-10086', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']

    for feat in tqdm(sparse_features):
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    for feat in tqdm(dense_features):
        mean = data[feat].mean()
        std = data[feat].std()
        data[feat] = (data[feat] - mean) / (std + 1e-12)

    print(data.shape)
    # data.head()

    train, valid = train_test_split(data, test_size=0.2, random_state=2020)
    print(train.shape, valid.shape)

    train_dataset = Data.TensorDataset(torch.LongTensor(train[sparse_features].values),
                                       torch.FloatTensor(train[dense_features].values),
                                       torch.FloatTensor(train['label'].values), )

    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=2048, shuffle=True)

    valid_dataset = Data.TensorDataset(torch.LongTensor(valid[sparse_features].values),
                                       torch.FloatTensor(valid[dense_features].values),
                                       torch.FloatTensor(valid['label'].values), )
    valid_loader = Data.DataLoader(dataset=valid_dataset, batch_size=4096, shuffle=False)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    cate_fea_nuniqs = [data[f].nunique() for f in sparse_features]
    model = DeepFM(cate_fea_nuniqs, nume_fea_size=len(dense_features))
    model.to(device)
    loss_fcn = nn.BCELoss()  # Loss函数
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)


    # 打印模型参数
    def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}


    print(get_parameter_number(model))


    # 定义日志（data文件夹下）
    def write_log(w):
        file_name = './' + datetime.date.today().strftime('%m%d') + "_{}.log".format("deepfm")
        t0 = datetime.datetime.now().strftime('%H:%M:%S')
        info = "{} : {}".format(t0, w)
        print(info)
        with open(file_name, 'a') as f:
            f.write(info + '\n')


    def train_and_eval(model, train_loader, valid_loader, epochs, device):
        best_auc = 0.0
        for _ in range(epochs):
            """训练部分"""
            model.train()
            print("Current lr : {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
            write_log('Epoch: {}'.format(_ + 1))
            train_loss_sum = 0.0
            start_time = time.time()
            for idx, x in enumerate(train_loader):
                cate_fea, nume_fea, label = x[0], x[1], x[2]
                cate_fea, nume_fea, label = cate_fea.to(device), nume_fea.to(device), label.float().to(device)
                pred = model(cate_fea, nume_fea).view(-1)
                loss = loss_fcn(pred, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.cpu().item()
                if (idx + 1) % 50 == 0 or (idx + 1) == len(train_loader):
                    write_log("Epoch {:04d} | Step {:04d} / {} | Loss {:.4f} | Time {:.4f}".format(
                        _ + 1, idx + 1, len(train_loader), train_loss_sum / (idx + 1), time.time() - start_time))
            scheduler.step()
            """推断部分"""
            model.eval()
            with torch.no_grad():
                valid_labels, valid_preds = [], []
                for idx, x in tqdm(enumerate(valid_loader)):
                    cate_fea, nume_fea, label = x[0], x[1], x[2]
                    cate_fea, nume_fea = cate_fea.to(device), nume_fea.to(device)
                    pred = model(cate_fea, nume_fea).reshape(-1).data.cpu().numpy().tolist()
                    valid_preds.extend(pred)
                    valid_labels.extend(label.cpu().numpy().tolist())
            cur_auc = roc_auc_score(valid_labels, valid_preds)
            if cur_auc > best_auc:
                best_auc = cur_auc
                torch.save(model.state_dict(), "./deepfm_best.pth")
            write_log('Current AUC: %.6f, Best AUC: %.6f\n' % (cur_auc, best_auc))


    train_and_eval(model, train_loader, valid_loader, 30, device)






