import sys
sys.path.append('/home/sci/zhichao.xu/Constrained-QFS')

import os
import time
import argparse
from tqdm import tqdm

import numpy as np
import torch

from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from datasets import load_dataset

from utils.utils import str2bool
from evaluation.eval_rouge import RougeEvaluator

class SummaryDataset(torch.utils.data.Dataset):
    def __init__(self, input_data, tokenizer, source_max_length, target_max_length):
        self.data = input_data
        self.tokenizer = tokenizer
        self.source_max_length = source_max_length
        self.target_max_length = target_max_length

    def __len__(self):
        return len(self.data[0])
    
    def __getitem__(self, index):
        ctext = self.data[0][index]
        target = self.data[1][index]

        tokenized_ctext = self.tokenizer(ctext, max_length=self.source_max_length, truncation=True, pad_to_max_length=True, return_tensors='pt')
        tokenized_target = self.tokenizer(target, max_length=self.target_max_length, truncation=True, pad_to_max_length=True, return_tensors='pt')

        source_ids = tokenized_ctext['input_ids'].squeeze()
        source_mask = tokenized_ctext['attention_mask'].squeeze()
        target_ids = tokenized_target['input_ids'].squeeze()
        target_mask = tokenized_target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_mask': target_mask.to(dtype=torch.long),
        }

def train(epoch, tokenizer, model, device, loader, optimizer, args):
    model.train()
    epoch_loss = 0

    pbar = tqdm(total=len(loader))
    for i, data in enumerate(loader):
        pbar.update(1)
        y = data['target_ids'].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype=torch.long)
        mask = data['source_mask'].to(device, dtype=torch.long)
    
        outputs = model(input_ids=ids, attention_mask=mask, decoder_input_ids=y_ids, labels=lm_labels)
        loss = outputs[0]
        epoch_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    pbar.close()
    print('Epoch loss: {}'.format(epoch_loss/len(loader)))

def validate(epoch, tokenizer, model, device, loader, args):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        pbar = tqdm(len(loader))
        for i, data in enumerate(loader):
            pbar.update(1)
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                min_length = args.min_tgt_length,
                max_length = args.max_tgt_length, 
                num_beams = 5,
                repetition_penalty = 1.0, 
                length_penalty = 2.5, 
                early_stopping = True
                )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]

            predictions.extend(preds)
            actuals.extend(target)
        pbar.close()
    return predictions, actuals


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment_root', type=str, required=True)
    
    parser.add_argument('--tokenizer', type=str, default='t5-base')
    parser.add_argument('--model', type=str, default='t5-base')

    # parser.add_argument('--')
    args = parser.parse_args()


    batch_size = 64
    tokenizer = AutoTokenizer.from_pretrained('t5-base')
    dataset_full = load_dataset('xsum')
    trainset = dataset_full['train']
    print(trainset)