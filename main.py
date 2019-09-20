from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchtext import data
import time
import os
import copy
from torch.utils.data import Dataset, DataLoader
from random import randrange
import torch.nn.functional as F

import torch
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM

import datahandler as dh 

dropout_prob = 0.3
hidden_size = 768
num_labels = 2548
state_dict_path = 'bert_dict_1.pth'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_seq_length = 256

class BertForSequenceClassification(nn.Module):
  
    def __init__(self, num_labels=2):
        super(BertForSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)
        nn.init.xavier_normal_(self.classifier.weight)
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        # _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        _,pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits

class text_dataset(Dataset):
    def __init__(self,x_y_list):
        self.x_y_list = x_y_list
        
    def __getitem__(self,index):
        
        tokenized_review = tokenizer.tokenize(self.x_y_list[0][index])
        
        # [NOTE] The truncation could be done from either side
        if len(tokenized_review) > max_seq_length:
            tokenized_review = tokenized_review[:max_seq_length]
            
        ids_review  = tokenizer.convert_tokens_to_ids(tokenized_review)

        padding = [0] * (max_seq_length - len(ids_review))
        
        ids_review += padding
        
        assert len(ids_review) == max_seq_length
        
        #print(ids_review)
        ids_review = torch.tensor(ids_review)
        
        # [DONE] [TODO] sentiment should be binary vector of skills labels
        sentiment = self.x_y_list[1][index] # color        
        list_of_labels = [torch.from_numpy(np.array(sentiment))]
        
        
        return ids_review, list_of_labels[0]
    
    def __len__(self):
        return len(self.x_y_list[0])

# DATA PROCESSING

batch_size = 32
dataloaders_dict = {}
dataloaders_dict['train'] = text_dataset(dh.dataloaders_dict['train'])
dataloaders_dict['val'] = text_dataset(dh.dataloaders_dict['val'])
'''skill_list = dh.get_skill_list()
train_iter, valid_iter = dh.load_dataset(batch_size=batch_size)
train_dl = dh.BatchWrapper(train_iter, 'description',skill_list )
valid_dl = dh.BatchWrapper(valid_iter, 'description', skill_list)
'''
dataset_sizes = {}
dataset_sizes['train'] = len(dataloaders_dict['train'][0])
dataset_sizes['val'] = len(dataloaders_dict['val'][0])

'''dataset_sizes['train'] = 15000
dataset_sizes['val'] = 5000
'''
'''dataloaders_dict['train'] = train_dl
dataloaders_dict['val'] = valid_dl'''

dataloaders_dict['train'] = DataLoader(dataloaders_dict['train'], batch_size=batch_size, shuffle=False)
dataloaders_dict['val'] = DataLoader(dataloaders_dict['val'], batch_size=batch_size, shuffle=False)


# dataloaders_dict['train'], dataloaders_dict['val'] = data.BucketIterator.splits((train_dl, valid_dl), batch_size=batch_size,sort_key=lambda x: len(x.text), repeat=False, shuffle=False)
# tokenized_text = tokenizer.tokenize(some_text)
# tokenizer.convert_tokens_to_ids(tokenized_text)

# [TODO] Add module for converting the data to required format


model = BertForSequenceClassification(num_labels)

model.load_state_dict(torch.load(state_dict_path))

# for p in model.bert.parameters():
#     p.requires_grad=False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# [TODO] Add data dictionary for iterating over data.

def train_model(model,  optimizer, scheduler,  num_epochs=25):
    since = time.time()
    print('starting')
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100
    model.to(device)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        scheduler.step()
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                # scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            
            # sentiment_corrects = 0
            idx = 0
            precision = []
            recall = []
            
            # Iterate over data.
            for inputs, sentiment in dataloaders_dict[phase]:
                #inputs = inputs
                #print(len(inputs),type(inputs),inputs)
                #inputs = torch.from_numpy(np.array(inputs)).to(device) 
                inputs = inputs.to(device) 
                sentiment = sentiment.type(torch.FloatTensor).to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    #print(inputs)
                    outputs = model(inputs)

                    outputs = F.sigmoid(outputs)
                    
                    loss = F.binary_cross_entropy(outputs, sentiment.to(device))
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                idx += 1
                outputs = outputs.cpu().detach().numpy()
                sentiment = sentiment.cpu().detach().numpy()
                predictions = np.array(outputs >= 0.5, dtype=np.int)
                corrects = np.array( np.array(predictions,dtype=np.int)  == np.array(sentiment, dtype=np.int), dtype=np.int )
                true_positives = (corrects*predictions).sum().item()
                total_positives_act = np.array(sentiment).sum().item()
                total_positives_pred = predictions.sum().item()
                precision_curr = true_positives/float(total_positives_pred+0.00001)
                recall_curr = true_positives/float(total_positives_act + 0.00001)

                precision.append(precision_curr)
                recall.append(recall_curr)

                print('Iter: {} Precision: {:.4f} Recall: {:.4f} Loss: {:.5f}'.format(idx, precision_curr, recall_curr, loss.item()))
                
                # sentiment_corrects += torch.sum(torch.max(outputs, 1)[1] == torch.max(sentiment, 1)[1])

                
            epoch_loss = running_loss / dataset_sizes[phase]

            
            # sentiment_acc = sentiment_corrects.double() / dataset_sizes[phase]
            recall_avg = np.mean(recall)*100
            precision_avg = np.mean(precision)*100
            print('{} total loss: {:.4f} '.format(phase,epoch_loss ))
            print('AVERAGE:: Precision {}, Recall: {}'.format(precision_avg, recall_avg))
            print('-'*69)
            print()
            # print('{} sentiment_acc: {:.4f}'.format(
            #     phase, sentiment_acc))

            if phase == 'val' and epoch_loss < best_loss:
                print('saving with loss of {}'.format(epoch_loss),
                      'improved over previous {}'.format(best_loss))
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'bert_model_full_train.pth')


        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(float(best_loss)))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# [DONE]  [TODO] Remove BERT model parameters from the training reign

lrlast = 0.01
lrmain = .00001
# lrmain = 0.0
optim1 = optim.Adam(
    [
        {"params":model.bert.parameters(),"lr": lrmain},
        {"params":model.classifier.parameters(), "lr": lrlast},       
   ])

optimizer_ft = optim1
scheduler = lr_scheduler.StepLR(optim1, step_size = 1, gamma = 0.5)
train_model(model, optim1, scheduler, 30)
