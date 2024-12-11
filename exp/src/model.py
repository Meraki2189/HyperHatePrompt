from torch.nn.functional import softmax, sigmoid
# from src.CrossmodalTransformer import MULTModel
# from src.StoG import *
# from src.GraphCAGE import *
from src.HypergraphLearning import *


# from src.HGNN import *


class GCN_CAPS_Model(nn.Module):
    def __init__(self, args, label_dim, t_in, a_in, v_in, T_t, T_a, T_v,
                 MULT_d,
                 vertex_num,
                 dim_capsule,
                 routing,
                 dropout):
        super(GCN_CAPS_Model, self).__init__()
        self.d_c = dim_capsule # 胶囊维度
        self.n = vertex_num # 节点数
        self.T_t = T_t # 输入序列的长度
        self.T_a = T_a
        self.T_v = T_v
        # self.T_p = T_p
        self.dropout = dropout

        # 构建超图
        self.HyperG = HyperedgeConstruction(args, self.d_c)
        # 超图卷积
        self.HyperGraphConv = HyperConv(args)
        # self.HyperGNN = HGNN(args, dim_capsule)
        # decode part
        self.fc1 = nn.Linear(in_features=4 * dim_capsule*2, out_features=dim_capsule*2)
        self.fc2 = nn.Linear(in_features=dim_capsule*2, out_features=label_dim)

    def forward(self, text, audio, video,prompt,  batch_size): #使用的是forward
        adjacency, nodes_list = self.HyperG(text, audio, video,prompt, batch_size)  # 超边扩展，返回邻接矩阵
        logits = self.HyperGraphConv(adjacency, nodes_list) # 超图卷积，返回最终节点表示
        output1 = torch.tanh(self.fc1(logits))
        output2 = F.dropout(output1, p=self.dropout, training=self.training)
        # preds = self.fc2(output2) * 10
        preds = softmax(self.fc2(output2),dim=1)
        return preds

        # 用HGNN
        # nodes_list, G = self.HyperG(nodes_t, nodes_a, nodes_v, batch_size)
        # logits = self.HyperGNN(nodes_list, G)
        # output1 = torch.tanh(self.fc3(logits))
        # preds = self.fc4(output1) * 10
        # return preds