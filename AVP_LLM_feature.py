import torch
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import numpy as np


def get_mask(seq, maxlength):
    return np.array([True] * len(seq) + [False] * (maxlength - len(seq)))


def get_bert(seq, tokenizer, model, maxlength):
    with torch.no_grad():
        len_seq = len(seq)
        seq = " ".join(seq)
        inputs = tokenizer.encode_plus(seq, max_length=maxlength + 2, padding='max_length', return_tensors="pt",
                                       return_attention_mask=True)
        input_ids = inputs["input_ids"].to(device)
        token_type_ids = inputs["token_type_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        last_hidden_states = torch.squeeze(outputs.last_hidden_state)
        indices_to_remove = [0, len_seq + 1] 
        mask = torch.ones(last_hidden_states.size(0), dtype=bool) 
        mask[indices_to_remove] = False
        last_hidden_states_process = last_hidden_states[mask]
        last_hidden_states_process = last_hidden_states_process.to("cpu").numpy()
        return last_hidden_states_process


def get_ESM(seq, tokenizer, model, maxlength):
    with torch.no_grad():
        inputs = tokenizer.encode_plus(seq, max_length=maxlength + 2, padding='max_length', return_tensors="pt",
                                       return_attention_mask=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = torch.squeeze(outputs.last_hidden_state)
        indices_to_remove = [0, len(seq) + 1] 
        mask = torch.ones(last_hidden_states.size(0), dtype=bool)
        mask[indices_to_remove] = False
        last_hidden_states_process = last_hidden_states[mask]
        last_hidden_states_process = last_hidden_states_process.to("cpu").numpy()
        return last_hidden_states_process



def read_xlsx(train_path, test_path):
    train_data = pd.read_excel(train_path)
    test_data = pd.read_excel(test_path)
    train_seq = train_data.iloc[:, 0].to_numpy()
    train_label = train_data.iloc[:, 1].to_numpy()
    test_seq = test_data.iloc[:, 0].to_numpy()
    test_label = test_data.iloc[:, 1].to_numpy()

    vectorized_len = np.vectorize(len)
    train_lengths = vectorized_len(train_seq)
    train_maxlength = np.max(train_lengths)
    test_lengths = vectorized_len(test_seq)
    test_maxlength = np.max(test_lengths)

    if train_maxlength > test_maxlength:
        maxlength = train_maxlength
    else:
        maxlength = test_maxlength
    return train_seq, train_label, test_seq, test_label, maxlength


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_path = "./data/AV_train.xlsx"
    test_path = "./data/AV_test.xlsx"
    model_path = '/prot_T5'
    train_save_path = "./data/LLM_feature/Prot-T5/Train"
    test_save_path = "./data/LLM_feature/Prot-T5/Test"

    tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=False)
    model = AutoModel.from_pretrained(model_path).to(device)

    train_seq, train_label, test_seq, test_label, maxlength = read_xlsx(train_path, test_path)
    train_mask = np.array([get_mask(i, maxlength) for i in train_seq])
    train_BERT_features = np.array([get_bert(i, tokenizer, model, maxlength) for i in train_seq])

    test_mask = np.array([get_mask(i, maxlength) for i in test_seq])
    test_BERT_features = np.array([get_bert(i, tokenizer, model, maxlength) for i in test_seq])

    np.savez(train_save_path, train_BERT_features=train_BERT_features, train_label=train_label, train_mask=train_mask)
    np.savez(test_save_path, test_BERT_features=test_BERT_features, test_label=test_label, test_mask=test_mask)
