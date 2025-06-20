import torch
import numpy as np
import pandas as pd
from AVP_model import Model, PeptideDataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef, roc_auc_score
import random
from transformers import AutoModel, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sed_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 计算评估指标
def compute_metrics(pred, label):
    tn, fp, fn, tp = confusion_matrix(label, pred).ravel()
    SN = tp / (tp + fn)
    SP = tn / (tn + fp)
    MCC = matthews_corrcoef(label, pred)
    ACC = accuracy_score(label, pred)
    AUC = roc_auc_score(label, pred)
    return ACC, MCC, SP, SN, AUC


def validing(model, valid_loader, BCEloss, alpha):
    model.eval()
    pred_list = []
    label_list = []
    valid_loss = 0
    with torch.no_grad():
        for ESM, manual, mask, label in valid_loader:
            ESM = ESM.to(device)
            manual = manual.to(device)
            mask = mask.to(device)
            label = label.to(device)
            label = label.unsqueeze(1)
            output, Contrastive_loss = model(ESM, manual, mask)
            BCE_loss = BCEloss(output, label)
            loss = alpha * Contrastive_loss + (1 - alpha) * BCE_loss
            valid_loss += loss
            pred_list.extend(np.round(output.cpu().detach().numpy()))
            label_list.extend(label.cpu().detach().numpy())
        ACC, MCC, SP, SN, AUC = compute_metrics(pred_list, label_list)
        return ACC, MCC, SP, SN, AUC, valid_loss / len(valid_loader)


def get_mask(seq, maxlength):
    return np.array([True] * len(seq) + [False] * (maxlength - len(seq)))


# 单独序列获得ESM，需要传入序列和最大长度
def get_ESM(seq, tokenizer, model, maxlength):
    with torch.no_grad():
        inputs = tokenizer.encode_plus(seq, max_length=maxlength + 2, padding='max_length', return_tensors="pt",
                                       return_attention_mask=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = torch.squeeze(outputs.last_hidden_state)
        indices_to_remove = [0, len(seq) + 1]  # 去除特殊字符的对应的embed ing，便于后续对齐操作
        mask = torch.ones(last_hidden_states.size(0), dtype=bool)  # 使用布尔掩码去除指定的索引
        mask[indices_to_remove] = False
        last_hidden_states_process = last_hidden_states[mask]
        last_hidden_states_process = last_hidden_states_process.to("cpu").numpy()
        return last_hidden_states_process


def get_BLOSUM62(seq):
    if len(seq) < 100:
        seq = seq + (100 - len(seq)) * "X"
    elif len(seq) > 100:
        seq = seq[:100]
    blosum62 = {
        'A': [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
        'R': [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],
        'N': [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],
        'D': [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],
        'C': [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
        'Q': [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],
        'E': [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],
        'G': [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],
        'H': [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],
        'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],
        'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],
        'K': [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],
        'M': [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],
        'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],
        'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],
        'S': [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],
        'T': [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],
        'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],
        'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],
        'V': [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],
        'X': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    }

    encodings = []
    for j in seq:
        encodings.append(blosum62[j])
    return np.array(encodings)


# 读取数据集，返回训练集seq和label，测试集seq和label，以及最大字符串长度
def read_xlsx(test_path):
    test_data = pd.read_excel(test_path)
    test_seq = test_data.iloc[:, 0].to_numpy()
    test_label = test_data.iloc[:, 1].to_numpy()
    return test_seq, test_label


if __name__ == "__main__":
    sed_seed(66)
    # 设置路径
    test_path = "./AV_test.xlsx"
    model_path = '/data/LLM/ESM2-650M'
    tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=False)
    model = AutoModel.from_pretrained(model_path).to(device)
    maxlength = 100
    test_seq, test_label = read_xlsx(test_path)

    test_mask = np.array([get_mask(i, maxlength) for i in test_seq])
    test_ESM_features = np.array([get_ESM(i, tokenizer, model, maxlength) for i in test_seq])
    test_BLOSUM62 = np.array([get_BLOSUM62(i) for i in test_seq])



    test_ESM_features = torch.tensor(test_ESM_features, dtype=torch.float32)
    test_label = torch.tensor(test_label, dtype=torch.float32)
    test_mask = torch.tensor(test_mask, dtype=torch.float32)
    test_manual_features = torch.tensor(test_BLOSUM62, dtype=torch.float32)

    batch_size = 64
    manual_feature_dim = 20
    ESM_feature_dim = 1280
    Contrastive_proj_dim = 512
    MLP_hidden_dim = 512
    alpha = 0.15

    test_dataset = PeptideDataset(test_ESM_features, test_manual_features, test_mask, test_label)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 初始化模型
    model = Model(manual_feature_dim, ESM_feature_dim, Contrastive_proj_dim, MLP_hidden_dim)
    model.to(device)

    # 加载训练好的模型参数
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    BCEloss = torch.nn.BCELoss()

    # 测试
    ACC, MCC, SP, SN, AUC, valid_loss = validing(model, test_loader, BCEloss, alpha)
    print(f'Test_ACC:{ACC}, Test_MCC:{MCC}, Test_SP:{SP}, Test_SN:{SN}, Test_AUC:{AUC}')
