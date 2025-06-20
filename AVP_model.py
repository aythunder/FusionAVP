import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SelfAttention(nn.Module):
    def __init__(self, hid_dim, dropout=0.2):
        super().__init__()

        self.hid_dim = hid_dim
        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        # query = key = value   [batch size, sent len, hid dim]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        energy = torch.matmul(Q, K.permute(0, 2, 1)) / self.scale
        expanded_mask = mask.unsqueeze(-1).expand(-1, -1, energy.shape[2])
        if mask is not None:
            energy = energy.masked_fill(expanded_mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))
        x = torch.matmul(attention, V)
        x = self.fc(x)
        return x

class ContrastiveModel(torch.nn.Module):
    def __init__(self, num_hidden: int, num_proj_hidden: int, tau: float = 0.5):
        super().__init__()
        self.tau: float = tau
        self.fc1 = nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, X1, X2, mean=True):
        # X1, X2: [batch_size, length, embedding]
        h1 = self.projection(X1)
        h2 = self.projection(X2)

        l1 = self.semi_loss(h1, h2)
        l2 = self.semi_loss(h2, h1)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        return torch.einsum('bld,bmd->blm', z1, z2)

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        refl_sim_diag = refl_sim.diagonal(dim1=-2, dim2=-1)
        between_sim_diag = between_sim.diagonal(dim1=-2, dim2=-1)

        return -torch.log(
            between_sim_diag
            / (refl_sim.sum(dim=-1) + between_sim.sum(dim=-1) - refl_sim_diag)
        ).sum(dim=-1)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
   
        x = torch.mean(x, dim=1)  # [batch_size, embedding]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x,


class Model(nn.Module):
    def __init__(self, manual_feature_dim: int, ESM_feature_dim: int, Contrastive_proj_dim: int,
                 MLP_hidden_dim: int = 2):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(manual_feature_dim, ESM_feature_dim)
        self.esm_Self_attentio_1 = SelfAttention(ESM_feature_dim)
        self.manual_Self_attention = SelfAttention(ESM_feature_dim)
        self.esm_Self_attentio_2 = SelfAttention(ESM_feature_dim)
        self.ContrastiveModel = ContrastiveModel(ESM_feature_dim, Contrastive_proj_dim)
        self.MLPClassifier = MLPClassifier(ESM_feature_dim, MLP_hidden_dim)

    def forward(self, ESM_feature, manual_feature, mask):
        manual_feature = self.fc1(manual_feature)
        fusion_1 = self.esm_Self_attentio_1(manual_feature, ESM_feature, ESM_feature, mask)
        fusion_2 = self.manual_Self_attention(fusion_1, manual_feature, manual_feature, mask)
        fusion_3 = self.esm_Self_attentio_2(fusion_2, ESM_feature, ESM_feature, mask)

        contrastive_loss = self.ContrastiveModel(fusion_1, fusion_3)
        x, x_1, x_2 = self.MLPClassifier(fusion_3)
        # return x, contrastive_loss, fusion_3, x_1, x_2
        return x, contrastive_loss

class Model_not_fusion(nn.Module):
    def __init__(self, manual_feature_dim: int, ESM_feature_dim: int,
                 MLP_hidden_dim: int = 2):
        super(Model_not_fusion, self).__init__()
        self.MLPClassifier = MLPClassifier(ESM_feature_dim + manual_feature_dim, MLP_hidden_dim)

    def forward(self, ESM_feature, manual_feature, mask):
        fusion_3 = torch.cat((ESM_feature, manual_feature), dim=2)
        x, x_1, x_2 = self.MLPClassifier(fusion_3)
        return x, fusion_3, x_1, x_2


class PeptideDataset(Dataset):
    def __init__(self, LLM_feature, manual_feature, mask, labels):
        self.LLM_feature = LLM_feature
        self.manual_feature = manual_feature
        self.mask = mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x1 = self.LLM_feature[idx]
        x2 = self.manual_feature[idx]
        x3 = self.mask[idx]
        y = self.labels[idx]
        return x1, x2, x3, y
