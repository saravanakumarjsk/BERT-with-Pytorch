import torch.utils.data
import numpy as np
import pandas as pd
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertAdam, BertConfig

device = torch.device('cuda')

def convert_lines(data, max_seq_len, tokenizer):
    longer = 0
    all_token = []
    for text in data:
        tokens = tokenizer.tokenize(text)
        if len(tokens)>max_seq_len:
            tokens = tokens[:max_seq_len]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"]) + [0] * (max_seq_len - len(tokens))
        all_token.append(one_token)
    return np.array(all_token)

MAX_SEQUENCE_LENGTH = 220
BATCH_SIZE = 32

BERT_MODEL_PATH = 'E:/bertmodel/Bert Pretrained Models/uncased_l-12_h-768_a-12/'
bert_config = BertConfig('E:/bertmodel/bert_inference/bert_config.json')

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None, do_lower_case=True)

df = pd.read_csv('test.csv')
df['comment_text'] = df['comment_text'].astype(str)
x_test = convert_lines(df['comment_text'].fillna('DUMMY_VALUE'), MAX_SEQUENCE_LENGTH, )

# create the model

model = BertForSequenceClassification(bert_config, num_labels=1)
model.load_state_dict(torch.load("E:/bertmodel/bert_inference/bert_pytorch.bin"))
for param in model.parameters():
    param.require_grad = False
model.eval()

test_pred = np.zeros(len(x_test))
test = torch.util.data.TensorDataset(torch.tensor(x_test, dtype=torch.long))
test_loader = torch.util.Data.DataLoader(test, batch_size=32, shuffle=False)
tk = tqdm(test_loader)

for i, (x_batch) in enumerate(tk):
    pred = model(x_batch.to(device), attention_mask=(x_batch > 0).to(device), labels=None)
    test_preds[i * 32:(i + 1) * 32] = pred[:, 0].detach().cpu().squeeze().numpy()
test_pred = torch.sigmoid(torch.tensor(test_preds)).numpy().ravel()




    pred = model(x_batch.to(device), attention_mask=(x_batch > 0).to(device), labels=None)
    test_preds[i * 32:(i + 1) * 32] = pred[:, 0].detach().cpu().squeeze().numpy()

test_pred = torch.sigmoid(torch.tensor(test_preds)).numpy().ravel()


output = pd.DataFrame(dict{'id':df['id'], 
                          'prediction':test_pred})


output.to_csv('output.csv', index=False)



















