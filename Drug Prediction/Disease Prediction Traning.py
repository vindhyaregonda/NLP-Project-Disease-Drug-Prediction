#!/usr/bin/env python
# coding: utf-8

# In[1]:


# from google.colab import drive
# drive.mount('/content/drive')
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ['CUDA_VISIBLE_DEVICES']='5'

# In[9]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader, random_split
import torch
torch.manual_seed(0)
class CustomDataset(Dataset):
    def __init__(self, csv_file, label_encoders):
        self.data = pd.read_csv(csv_file)
        self.data = self.data[:].dropna()
        self.text = self.data['review'].tolist()
        self.target1 = self.data['drug'].tolist()
        self.target2 = self.data['condition'].tolist()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # Retrieve label encoders from the provided dictionary
        self.label_encoder1 = label_encoders['target1']
        self.label_encoder2 = label_encoders['target2']

        # Encode target variables using the retrieved label encoders
        self.target1_encoded = self.label_encoder1.transform(self.target1)
        self.target2_encoded = self.label_encoder2.transform(self.target2)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.text[idx]
        target1 = self.target1_encoded[idx]
        target2 = self.target2_encoded[idx]

        # Tokenize the text
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'target1': target1,
            'target2': target2,
            'original_target1': self.target1[idx],
            'original_target2': self.target2[idx]
        }

csv_file = '/media/nas_mount/debnath/Thesis/BERT_V/df_train_20.csv'
# csv_file = '/content/drive/MyDrive/NLP Project-24/Drug_Map/fdf.csv'  # Replace 'your_data.csv' with the path to your CSV file

# Create label encoders for target variables
label_encoders = {}
label_encoders['target1'] = LabelEncoder()
label_encoders['target2'] = LabelEncoder()

# Fit label encoders on target variables
data = pd.read_csv(csv_file)
label_encoders['target1'].fit(data['drug'])
label_encoders['target2'].fit(data['condition'])

# Create dataset with label encoders
dataset = CustomDataset(csv_file, label_encoders)


# In[4]:


# import pandas as pd
# pd.read_csv("/content/drive/MyDrive/NLP Project-24/Drug_Map/fdf.csv")


# In[10]:


train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

# Divide the dataset into training and validation sets
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Example usage of dataloaders for training and validation
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)


# In[ ]:


# for i in train_dataloader:
#   print(i)


# In[11]:


import torch
import torch.nn as nn
from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self, num_classes1, num_classes2):
        super(BertClassifier, self).__init__()
        # self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('/media/nas_mount/debnath/Thesis/BERT_V/my_model')
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, num_classes1)
        

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)

        # Use pooled_output for final classification
        logits1 = self.fc1(pooled_output)
      

        return logits1

# Example usage:
num_classes1 = len(label_encoders['target1'].classes_)
num_classes2 = len(label_encoders['target2'].classes_)
model = BertClassifier(num_classes1, num_classes2)


# In[12]:


import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# Define loss function
criterion = nn.CrossEntropyLoss()


# In[13]:


# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# Define loss function
criterion = nn.CrossEntropyLoss()


# In[18]:


def calculate_metrics(preds, targets):
    # If preds is a 2D tensor with shape (batch_size, num_classes), then use argmax along dim=1
    if preds.dim() == 2:
        preds = preds.argmax(dim=1).cpu().numpy()
    # If preds is a 1D tensor with shape (batch_size,), then no need for argmax
    elif preds.dim() == 1:
        preds = preds.cpu().numpy()
    else:
        raise ValueError("Unsupported shape for preds tensor")

    targets = targets.cpu().numpy()
    accuracy = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='macro')
    return accuracy, f1


# In[22]:


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0.0
    all_preds1 = []
    all_targets1 = []
    all_preds2 = []
    all_targets2 = []

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Training')
    for batch_idx, batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        target1 = batch['target1'].type(torch.LongTensor).to(device)
        # target2 = batch['target2'].type(torch.LongTensor).to(device)

        optimizer.zero_grad()
        logits1 = model(input_ids, attention_mask)
        loss1 = criterion(logits1, target1)
        # loss2 = criterion(logits2, target2)
        loss = loss1
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # Calculate metrics for target1
        all_preds1.extend(logits1.argmax(dim=1).cpu().numpy())
        all_targets1.extend(target1.cpu().numpy())

        # Calculate metrics for target2
        # all_preds2.extend(logits2.argmax(dim=1).cpu().numpy())
        # all_targets2.extend(target2.cpu().numpy())

        progress_bar.set_postfix({'loss': epoch_loss / (batch_idx + 1)})


    # Calculate metrics for target1
    accuracy1, f1_1 = calculate_metrics(torch.tensor(all_preds1), torch.tensor(all_targets1))

    # Calculate metrics for target2
    # accuracy2, f1_2 = calculate_metrics(torch.tensor(all_preds2), torch.tensor(all_targets2))

    return epoch_loss / len(dataloader), accuracy1, f1_1#, accuracy2, f1_2


# In[23]:


def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0.0
    all_preds1 = []
    all_targets1 = []
    all_preds2 = []
    all_targets2 = []

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Validation')
    for batch_idx, batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        target1 = batch['target1'].type(torch.LongTensor).to(device)
        

        logits1 = model(input_ids, attention_mask)
        loss1 = criterion(logits1, target1)
        # loss2 = criterion(logits2, target2)
        loss = loss1 #+ loss2

        epoch_loss += loss.item()

        # Calculate metrics for target1
        all_preds1.extend(logits1.argmax(dim=1).cpu().numpy())
        all_targets1.extend(target1.cpu().numpy())

        # Calculate metrics for target2
        # all_preds2.extend(logits2.argmax(dim=1).cpu().numpy())
        # all_targets2.extend(target2.cpu().numpy())

        progress_bar.set_postfix({'loss': epoch_loss / (batch_idx + 1)})

    # Calculate metrics for target1
    accuracy1, f1_1 = calculate_metrics(torch.tensor(all_preds1), torch.tensor(all_targets1))

    # Calculate metrics for target2
    # accuracy2, f1_2 = calculate_metrics(torch.tensor(all_preds2), torch.tensor(all_targets2))

    return epoch_loss / len(dataloader), accuracy1, f1_1


# In[24]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 3  # Define the number of epochs
model = model.to(device)
for epoch in range(num_epochs):
    train_loss, train_accuracy1, train_f1_1 = train_epoch(model, train_dataloader, optimizer, criterion, device)
    val_loss, val_accuracy1, val_f1_1 = evaluate(model, val_dataloader, criterion, device)

    print(f'Epoch {epoch+1}/{num_epochs}:')
    print(f'Train Loss: {train_loss:.4f}, Train Accuracy Target 1: {train_accuracy1:.4f}, Train F1 Score Target 1: {train_f1_1:.4f}') #, Train Accuracy Target 2: {train_accuracy2:.4f}, Train F1 Score Target 2: {train_f1_2:.4f}')
    print(f'Val Loss: {val_loss:.4f}, Val Accuracy Target 1: {val_accuracy1:.4f}, Val F1 Score Target 1: {val_f1_1:.4f}')#, Val Accuracy Target 2: {val_accuracy2:.4f}, Val F1 Score Target 2: {val_f1_2:.4f}')


# In[ ]:


# model.save_pretrained('./model_uncased')
model_path = "./model_solo1_20.pth"
torch.save(model.state_dict(), model_path)

print("Model saved successfully!")


# In[ ]:


# def inference(model, dataloader, label_encoders, device):
#     model.eval()
#     all_preds1 = []
#     all_preds2 = []
#     all_targets1 = []
#     all_targets2 = []

#     for batch in dataloader:
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         target1 = batch['target1'].to(device)
#         target2 = batch['target2'].to(device)

#         logits1, logits2 = model(input_ids, attention_mask)

#         all_preds1.extend(logits1.argmax(dim=1).cpu().numpy())
#         all_preds2.extend(logits2.argmax(dim=1).cpu().numpy())
#         all_targets1.extend(target1.cpu().numpy())
#         all_targets2.extend(target2.cpu().numpy())

#     # Reverse label encoding
#     label_encoder1 = label_encoders['target1']
#     label_encoder2 = label_encoders['target2']
#     original_preds1 = label_encoder1.inverse_transform(all_preds1)
#     original_preds2 = label_encoder2.inverse_transform(all_preds2)
#     original_targets1 = label_encoder1.inverse_transform(all_targets1)
#     original_targets2 = label_encoder2.inverse_transform(all_targets2)

#     return original_preds1, original_preds2, original_targets1, original_targets2

# # Perform inference
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# preds1, preds2, targets1, targets2 = inference(model, val_dataloader, label_encoders, device)

# # Analysis of predictions
# for pred1, pred2, target1, target2 in zip(preds1, preds2, targets1, targets2):
#     print(f'Predicted target1: {pred1}, Original target1: {target1}')
#     print(f'Predicted target2: {pred2}, Original target2: {target2}')
#     print()  # Add a newline for readability

