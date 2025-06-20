import pandas as pd
import numpy as np


def get_AAC(seq):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aac_dict = {aa: 0 for aa in amino_acids}
    for aa in seq:
        if aa in aac_dict:
            aac_dict[aa] += 1
    aac_features = [aac_dict[aa] / len(seq) for aa in amino_acids]
    return np.array(aac_features)


def get_DPC(seq):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    dpc_dict = {aa1 + aa2: 0 for aa1 in amino_acids for aa2 in amino_acids}
    for i in range(len(seq) - 1):
        dipeptide = seq[i:i + 2]  # 取当前和下一个氨基酸组成的二肽
        if dipeptide in dpc_dict:
            dpc_dict[dipeptide] += 1
    dpc_features = [dpc_dict[dipeptide] / (len(seq) - 1) for dipeptide in dpc_dict]
    return np.array(dpc_features)


def get_AAindex(seq):
    if len(seq) < 100:
        seq = seq + (100 - len(seq)) * "X"
    elif len(seq) > 100:
        seq = seq[:100]
    AAindex = {'A': [89.3, 1.43, 9.36, 0.96, 16.0, 7.9, 0.5, 0.0, 9.25, 0.92, 154.33, -0.04],
               'C': [102.5, 0.94, 2.56, 0.42, 168.0, 1.9, 0.0, 0.0, 1.07, 1.16, 219.79, -0.38],
               'D': [114.4, 0.92, 0.94, 0.42, -78.0, 5.5, 0.0, -1.0, 3.89, 0.48, 194.91, 0.19],
               'E': [138.8, 1.67, 0.94, 0.53, -106.0, 7.1, 0.0, -1.0, 4.8, 0.61, 223.16, 0.23],
               'F': [190.8, 1.19, 10.99, 0.59, 189.0, 3.9, 2.5, 0.0, 6.36, 1.25, 204.74, -0.38],
               'G': [63.8, 0.46, 6.17, 0.0, -13.0, 7.1, 0.0, 0.0, 8.51, 0.61, 127.9, 0.09],
               'H': [157.5, 0.98, 0.47, 0.57, 50.0, 2.1, 0.5, 0.0, 1.88, 0.93, 242.54, -0.04],
               'I': [163.0, 1.04, 13.73, 0.84, 151.0, 5.2, 1.8, 0.0, 6.47, 1.81, 233.21, -0.34],
               'K': [165.1, 1.27, 0.58, 0.73, -141.0, 6.7, 0.0, 1.0, 3.5, 0.7, 300.46, 0.33],
               'L': [163.1, 1.36, 16.64, 0.92, 145.0, 8.6, 1.8, 0.0, 10.94, 1.3, 232.3, -0.37],
               'M': [165.8, 1.53, 3.93, 0.86, 124.0, 2.4, 1.3, 0.0, 3.14, 1.19, 202.65, -0.3],
               'N': [122.4, 0.64, 2.31, 0.39, -74.0, 4.0, 0.0, 0.0, 3.71, 0.6, 207.9, 0.13],
               'P': [121.6, 0.49, 1.96, -2.5, -20.0, 5.3, 0.0, 0.0, 4.36, 0.4, 179.93, 0.19],
               'Q': [146.9, 1.22, 1.14, 0.8, -73.0, 4.4, 0.0, 0.0, 3.17, 0.95, 235.51, 0.14],
               'R': [190.3, 1.18, 0.27, 0.77, -70.0, 4.9, 0.0, 1.0, 3.96, 0.93, 341.01, 0.07],
               'S': [94.2, 0.7, 5.58, 0.53, -70.0, 6.6, 0.0, 0.0, 6.26, 0.82, 174.06, 0.12],
               'T': [119.6, 0.78, 4.68, 0.54, -38.0, 5.3, 0.4, 0.0, 5.66, 1.12, 205.8, 0.03],
               'V': [138.2, 0.98, 12.43, 0.63, 123.0, 6.8, 1.5, 0.0, 7.55, 1.81, 207.6, -0.29],
               'W': [226.4, 1.01, 2.2, 0.58, 145.0, 1.2, 3.4, 0.0, 2.22, 1.54, 237.01, -0.33],
               'Y': [194.6, 0.69, 3.13, 0.72, 53.0, 3.1, 2.3, 0.0, 3.28, 1.53, 229.15, -0.29],
               'X': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    AA = []
    for j in seq:
        AA.append(AAindex[j])
    return np.array(AA)


def get_one_hot(seq):
    if len(seq) < 100:
        seq = seq + (100 - len(seq)) * "X"
    elif len(seq) > 100:
        seq = seq[:100]
    one_hot = {
        'A': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'C': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'D': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'E': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'F': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'G': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'H': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'I': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'K': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'L': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'M': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'N': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        'P': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        'Q': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        'R': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        'S': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        'T': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        'V': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        'W': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        'Y': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        'X': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    }

    encodings = []
    for j in seq:
        encodings.append(one_hot[j])
    return np.array(encodings)


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


def get_CTDC(seq):
    groups = [
        {'hydrophobicity_PRAM900101': 'RKEDQN', 'normwaalsvolume': 'GASTPDC', 'polarity': 'LIFWCMVY'},
        {'hydrophobicity_PRAM900101': 'GASTPHY', 'normwaalsvolume': 'NVEQIL', 'polarity': 'PATGS'},
        {'hydrophobicity_PRAM900101': 'CLVIMFW', 'normwaalsvolume': 'MHKFRYW', 'polarity': 'HQRKNED'}
    ]
    properties = ['hydrophobicity_PRAM900101', 'normwaalsvolume', 'polarity']
    features = []
    for prop in properties:
        c1 = sum(seq.count(aa) for aa in groups[0][prop]) / len(seq)
        c2 = sum(seq.count(aa) for aa in groups[1][prop]) / len(seq)
        c3 = 1 - c1 - c2
        # 将比例添加到特征列表
        features.extend([c1, c2, c3])
    return np.array(features)


def get_CTDT(seq):
    # 定义分组，基于不同的物理化学属性分类
    groups = [
        {'hydrophobicity_PRAM900101': 'RKEDQN', 'normwaalsvolume': 'GASTPDC', 'polarity': 'LIFWCMVY'},
        {'hydrophobicity_PRAM900101': 'GASTPHY', 'normwaalsvolume': 'NVEQIL', 'polarity': 'PATGS'},
        {'hydrophobicity_PRAM900101': 'CLVIMFW', 'normwaalsvolume': 'MHKFRYW', 'polarity': 'HQRKNED'}
    ]

    # 定义需要计算的属性
    properties = ['hydrophobicity_PRAM900101', 'normwaalsvolume', 'polarity']

    # 初始化特征列表
    features = []

    # 遍历每个属性
    for prop in properties:
        # 映射每个氨基酸到相应的组（1, 2, 3）
        group_mapping = {}
        for i, group in enumerate(groups):
            for aa in group[prop]:
                group_mapping[aa] = i + 1

        # 计算转变频率
        transitions = {'12': 0, '13': 0, '23': 0}
        seq_groups = [group_mapping[aa] for aa in seq if aa in group_mapping]  # 将序列转换为组编号

        # 计算每个转变的次数
        for i in range(len(seq_groups) - 1):
            pair = f"{seq_groups[i]}{seq_groups[i + 1]}"
            if pair in transitions:
                transitions[pair] += 1

        # 归一化处理，避免除以零
        total_transitions = sum(transitions.values())
        if total_transitions > 0:
            for key in transitions:
                transitions[key] /= total_transitions
        else:
            for key in transitions:
                transitions[key] = 0  # 如果总转变次数为0，将值设为0

        # 将计算的转变特征添加到特征列表
        features.extend([transitions['12'], transitions['13'], transitions['23']])

    return np.array(features)


def get_CTDD(seq):
    # 定义分组，基于不同的物理化学属性分类
    groups = [
        {'hydrophobicity_PRAM900101': 'RKEDQN', 'normwaalsvolume': 'GASTPDC', 'polarity': 'LIFWCMVY'},
        {'hydrophobicity_PRAM900101': 'GASTPHY', 'normwaalsvolume': 'NVEQIL', 'polarity': 'PATGS'},
        {'hydrophobicity_PRAM900101': 'CLVIMFW', 'normwaalsvolume': 'MHKFRYW', 'polarity': 'HQRKNED'}
    ]

    # 定义需要计算的属性
    properties = ['hydrophobicity_PRAM900101', 'normwaalsvolume', 'polarity']

    # 初始化特征列表
    features = []

    # 遍历每个属性
    for prop in properties:
        # 映射每个氨基酸到相应的组（1, 2, 3）
        group_positions = {1: [], 2: [], 3: []}  # 存储每组氨基酸的位置
        for i, aa in enumerate(seq):
            for group_index, group in enumerate(groups, 1):
                if aa in group[prop]:
                    group_positions[group_index].append(i + 1)  # 记录位置，1-based index

        # 计算每组在0%、25%、50%、75%、100%的位置上的分布
        for group_index in range(1, 4):
            positions = group_positions[group_index]
            if positions:
                features.extend([
                    positions[0] / len(seq),  # 0% 第一个出现的位置
                    positions[int(0.25 * (len(positions) - 1))] / len(seq),  # 25%
                    positions[int(0.5 * (len(positions) - 1))] / len(seq),  # 50%
                    positions[int(0.75 * (len(positions) - 1))] / len(seq),  # 75%
                    positions[-1] / len(seq)  # 100% 最后一个出现的位置
                ])
            else:
                # 如果某组没有氨基酸，则特征全为0
                features.extend([0, 0, 0, 0, 0])

    return np.array(features)


def get_GAAC(seq):
    # 定义氨基酸分组
    group = {
        'alphatic': 'GAVLMI',
        'aromatic': 'FYW',
        'positivecharge': 'KRH',
        'negativecharge': 'DE',
        'uncharge': 'STCPNQ'
    }

    # 初始化一个字典来存储每组的计数
    group_counts = {key: 0 for key in group}

    # 计算每个组的氨基酸数量
    for aa in seq:
        for key, aas in group.items():
            if aa in aas:
                group_counts[key] += 1

    # 计算每个组的比例
    total_aa = len(seq)
    features = [group_counts[key] / total_aa for key in group]

    return features


def get_GDPC(seq):
    # 定义氨基酸分组
    group = {
        'alphatic': 'GAVLMI',
        'aromatic': 'FYW',
        'positivecharge': 'KRH',
        'negativecharge': 'DE',
        'uncharge': 'STCPNQ'
    }

    # 创建一个二肽组合字典
    group_keys = list(group.keys())
    dipeptide_combinations = {f'{g1}-{g2}': 0 for g1 in group_keys for g2 in group_keys}

    # 映射每个氨基酸到它的组
    aa_to_group = {}
    for group_name, aas in group.items():
        for aa in aas:
            aa_to_group[aa] = group_name

    # 遍历序列中的二肽
    for i in range(len(seq) - 1):
        aa1, aa2 = seq[i], seq[i + 1]
        if aa1 in aa_to_group and aa2 in aa_to_group:
            group1 = aa_to_group[aa1]
            group2 = aa_to_group[aa2]
            dipeptide_combinations[f'{group1}-{group2}'] += 1

    # 计算比例（归一化）
    total_dipeptides = sum(dipeptide_combinations.values())
    if total_dipeptides > 0:
        for key in dipeptide_combinations:
            dipeptide_combinations[key] /= total_dipeptides
    else:
        for key in dipeptide_combinations:
            dipeptide_combinations[key] = 0  # 如果没有二肽，设置比例为0

    # 转换为特征向量
    features = list(dipeptide_combinations.values())
    return features


# 读取数据集，返回训练集和测试集seq
def read_xlsx(train_path, test_path):
    train_data = pd.read_excel(train_path)
    test_data = pd.read_excel(test_path)
    train_seq = train_data.iloc[:, 0].to_numpy()
    test_seq = test_data.iloc[:, 0].to_numpy()

    return train_seq, test_seq


if __name__ == "__main__":
    train_path = "./data/AV_train.xlsx"
    test_path = "./data/AV_test.xlsx"

    train_save_path = "./data/Mannual_feature/AAindex/Train"
    test_save_path = "./data/Mannual_feature/AAindex/Test"

    train_seq, test_seq = read_xlsx(train_path, test_path)


    # train_GAAC = np.array([get_GAAC(i) for i in train_seq])
    # train_GDPC = np.array([get_GDPC(i) for i in train_seq])
    # train_GAACom_features = np.concatenate((train_GAAC, train_GDPC), axis=1)
    # test_GAAC = np.array([get_GAAC(i) for i in test_seq])
    # test_GDPC = np.array([get_GDPC(i) for i in test_seq])
    # test_GAACom_features = np.concatenate((test_GAAC, test_GDPC), axis=1)

    # train_CTDC = np.array([get_CTDC(i) for i in train_seq])
    # train_CTDT = np.array([get_CTDT(i) for i in train_seq])
    # train_CTDD = np.array([get_CTDD(i) for i in train_seq])
    # train_SD_features = np.concatenate((train_CTDC, train_CTDT, train_CTDD), axis=1)
    # test_CTDC = np.array([get_CTDC(i) for i in test_seq])
    # test_CTDT = np.array([get_CTDT(i) for i in test_seq])
    # test_CTDD = np.array([get_CTDD(i) for i in test_seq])
    # test_SD_features = np.concatenate((test_CTDC, test_CTDT, test_CTDD), axis=1)
    #
    train_AAindex_features = np.array([get_AAindex(i) for i in train_seq])
    test_AAindex_features = np.array([get_AAindex(i) for i in test_seq])

    np.savez(train_save_path, train_AAindex_features=train_AAindex_features)
    np.savez(test_save_path, test_AAindex_features=test_AAindex_features)
