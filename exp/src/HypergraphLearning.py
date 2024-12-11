import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
from scipy.sparse import coo_matrix


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

# 输入稀疏矩阵的超图卷积
class HyperConv(nn.Module):
    def __init__(self, args):
        super(HyperConv, self).__init__()
        self.layers = args.layers

    def forward(self, adjacency, embedding):
        item_embeddings = embedding #初始节点嵌入
        item_embedding_layer0 = item_embeddings
        final = item_embedding_layer0.unsqueeze(0)
        for i in range(self.layers):
            item_embeddings = torch.sparse.mm(trans_to_cuda(adjacency), item_embeddings)# 稀疏矩阵乘法
            final=torch.cat([final, item_embeddings.unsqueeze(0)], dim=0)
        item_embeddings = torch.sum(final, 0) / (self.layers+1) #最终的节点嵌入为几层的输出取平均
        # data_size = int(item_embeddings.shape[0]/3)
        # embedding_t, embedding_a, embedding_v = torch.split(item_embeddings, data_size, dim=0)
        # logits_embeddings = torch.cat([embedding_t, embedding_a, embedding_v], dim=1)
        data_size = int(item_embeddings.shape[0] / 4)
        embedding_t, embedding_a, embedding_v, embedding_p= torch.split(item_embeddings, data_size, dim=0)
        logits_embeddings = torch.cat([embedding_t, embedding_a, embedding_v, embedding_p], dim=1)
        return logits_embeddings


# 超边扩展
class HyperedgeConstruction(nn.Module):
    def __init__(self, args, dim_capsule):
        super(HyperedgeConstruction, self).__init__()
        # self.emb_size = dim_capsule * 2
        self.emb_size = dim_capsule
        self.k_2 = args.k_2
        self.d_c = dim_capsule

    def dfs_expand(self, H, h_new, idx_max, H_hash):
        torch.clamp_max(h_new, max=1.0)
        h_new_hash = hash(str(h_new))
        if h_new_hash not in H_hash:
            h_new_1 = h_new.unsqueeze(1)
            H = torch.cat((H, h_new_1), dim=1)
            H_hash.append(h_new_hash)
        for i in idx_max:
            self.dfs_expand(H, h_new + H[:, i], idx_max, H_hash)

    def forward(self, nodes_t, nodes_a, nodes_v,nodes_p, batch_size):
        # nodes_num = len(nodes_t)*3
        # nodes_list = torch.cat([nodes_t, nodes_a, nodes_v], 0)
        nodes_num = len(nodes_t) * 4
        nodes_list = torch.cat([nodes_t, nodes_a, nodes_v, nodes_p], 0)
        H = torch.zeros([nodes_num, batch_size], dtype = torch.float32) #[N,M]

        # 初始化关联矩阵H，同一个样本三个模态的节点为一条超边
        for i in range(batch_size):
            # H[i][i], H[i+batch_size][i], H[i+batch_size*2][i] = 1.0, 1.0, 1.0
            H[i][i], H[i + batch_size][i], H[i + batch_size * 2][i], H[i + batch_size * 3][i] = 1.0, 1.0, 1.0, 1.0

        B = torch.sum(H, dim=0)
        B = torch.diag(B) # 边的度
        # D = torch.sum(H, dim=1)
        # D = torch.diag(D) # 点的度
        B_I = B.inverse() # 求逆矩阵
        H_T = H.transpose(1, 0)

        # 计算超边的表示，这里没考虑节点权重，超边权重都为1
        hyperedges_list = torch.mm(torch.mm(B_I, H_T), nodes_list)

        # 计算曼哈顿距离（L1范数）/p=2也可以求欧氏距离
        list_dist = torch.norm(hyperedges_list[:, None] - hyperedges_list, dim=2, p=1)

        # 计算初始超边对应关联矩阵列的哈希值
        H_hash = []
        for i in range(batch_size):
            H_hash.append(hash(str(H[:, i])))

        # 1.为每条超边选择最可能扩展的k_2条超边
        k = list_dist.size()[0]
        if (k < self.k_2):
            self.k_2 = k
        list_dist_sort, idx_sort = torch.sort(list_dist, descending=True)  # descending为False，升序，为True，降序
        idx_max = idx_sort[:, :self.k_2]  # 截取排序后每行最大的k_2个元素的序号
        # 1 广度搜索
        # 根据选出的k_2条超边建立新的超边，即更新关联矩阵
        for i in range(batch_size):
            h_new = H[:, i]
            for j in range(self.k_2):
                h_new = h_new + H[:, idx_max[i, j]]
            torch.clamp_max(h_new, max=1.0)
            h_new_hash = hash(str(h_new))
            if (h_new_hash in H_hash) is False:  # 判断是否为重复超边
                h_new = h_new.unsqueeze(1)
                H = torch.cat((H, h_new), dim=1)

        # 2 深度搜索
        # for i in range(batch_size):
        #     h_new = H[:, i]
        #     self.dfs_expand(H, h_new, idx_max[i], H_hash)

        #3 混合搜索
        # 混合扩展
        # for i in range(batch_size):
        #     h_new = H[:, i]
        #     h_new_combine = h_new
        #     h_original_hash = hash(str(h_new))
        #     H_hash.append(h_original_hash)
        #
        #     for j in range(self.k_2):
        #         h_combine = h_new + H[:, idx_max[i, j]]
        #         torch.clamp_max(h_combine, max=1.0)
        #         h_combine_hash = hash(str(h_combine))
        #
        #         if h_combine_hash not in H_hash:
        #             h_combine = h_combine.unsqueeze(1)
        #             H = torch.cat((H, h_combine), dim=1)
        #             H_hash.append(h_combine_hash)
        #
        #             if h_combine_hash == h_original_hash:
        #                 h_new_combine = h_combine
        #
        #     H[:, i] = h_new_combine

        # # 消融实验：把已经被扩展的超边列清零
        # for i in range(batch_size):
        #     for j in range(self.k_2):
        #         H[:, idx_max[i, j]] = 0
        # # 将这些0列删除，只留下不为0的列
        # valid_cols = []
        # for col_idx in range(H.size(1)):
        #     if not torch.all(H[:, col_idx] == 0):
        #         valid_cols.append(col_idx)
        # H = H[:, valid_cols]

        # # 2.选择扩展可能性大于k_2的超边(确定阈值)
        # k = list_dist.size()[0]
        # mask = torch.zeros(k, k)
        # list_dist = torch.where(list_dist > self.k_2, list_dist, mask)
        # idx_value = torch.nonzero(list_dist)
        # for i in range(idx_value.size()[0]):
        #     h_new = H[:, idx_value[i, 0]] + H[:, idx_value[i, 1]]
        #     h_new_hash = hash(str(h_new))
        #     if (h_new_hash in H_hash) is False:  # 判断是否为重复超边
        #          h_new = h_new.unsqueeze(1)
        #          H = torch.cat((H, h_new), dim=1)

        # 计算邻接矩阵（稀疏矩阵），为了便于调用超图卷积方法
        row, col = torch.nonzero(H, as_tuple=True)
        data = torch.ones(row.shape).detach().cpu()
        row = row.detach().cpu().numpy()
        col = col.detach().cpu().numpy()
        H_coo = coo_matrix((data, (row, col)), shape=H.shape, dtype=np.float32)
        H_T_coo = H_coo.T
        BH_T = H_T_coo.T.multiply(1.0 / H_T_coo.sum(axis=1).reshape(1, -1))  # [N,M]
        BH_T = BH_T.T  # [M,N]
        DH = H_coo.T.multiply(1.0 / H_coo.sum(axis=1).reshape(1, -1))  # [M,N]
        DH = DH.T  # [N,M]
        DHBH_T = np.dot(DH, BH_T)  # [N,N]

        adjacency = DHBH_T.tocoo()
        # 将邻接矩阵从coo_matrix转化为tensor
        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adjacency.shape
        adjacency = trans_to_cuda(torch.sparse.FloatTensor(i, v, torch.Size(shape)))
        return adjacency, nodes_list
