import sys
sys.path.append('/home/sci/zhichao.xu/Constrained-QFS')

import os
import time
import argparse
import random
# random.seed(42)
import logging

from tqdm import tqdm
import numpy as np

import torch
# torch.manual_seed(42)

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from utils.utils import str2bool
from utils.input_utils import format_input, read_data_tsv
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

def val_targets(tokenizer, loader, device='cpu'):
    actuals = []
    for i, data in enumerate(loader):
        y = data['target_ids'].to(device, dtype = torch.long)
        target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]
        actuals.extend(target)
    return actuals

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
                num_beams = args.beam_size,
                repetition_penalty = 1.0, 
                length_penalty = 2.5, 
                early_stopping = True
                )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            # preds = [tokenizer.decode(g) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]

            predictions.extend(preds)
            actuals.extend(target)
        pbar.close()
    return predictions, actuals


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_dir', type=str, default='/home/sci/zhichao.xu/cg_dataset/dbpedia_processed')
    parser.add_argument('--experiment_root', type=str, required=True)
    parser.add_argument('--output_file', type=str)

    parser.add_argument('--tokenizer', type=str, default='t5-base')
    parser.add_argument('--model_name_or_path', type=str, default='t5-base')

    parser.add_argument('--source_max_length', type=int, default=150)
    parser.add_argument('--max_tgt_length', type=int, default=48)
    parser.add_argument('--min_tgt_length', type=int, default=5)

    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument(
        '--input_format', 
        type=str, 
        default='t5-summarize',
        choices=['bart-concat', 't5-summarize']
    )
    parser.add_argument('--valid_batch_size', type=int, default=4)
    parser.add_argument('--beam_size', type=int, default=5)

    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--save_ckpt', action="store_true")

    args = parser.parse_args()
    for k,v in vars(args).items():
        print(str(k) + ': ' + str(v))

    train_queries, train_contexts, train_targets = read_data_tsv(os.path.join(args.input_dir, 'train_triples.tsv'))
    valid_queries, valid_contexts, valid_targets = read_data_tsv(os.path.join(args.input_dir, 'test_triples.tsv'))

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    train_inputs = []
    for i, query in enumerate(train_queries):
        train_inputs.append(format_input(query, train_contexts[i], args.input_format))
    train_inputs = [train_inputs, train_targets]

    valid_inputs = []
    for i, query in enumerate(valid_queries):
        valid_inputs.append(format_input(query, valid_contexts[i], args.input_format))
    valid_inputs = [valid_inputs, valid_targets]

    trainset = SummaryDataset(train_inputs, tokenizer, args.source_max_length, args.max_tgt_length)
    validset = SummaryDataset(valid_inputs, tokenizer, args.source_max_length, args.max_tgt_length)
    train_loader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=args.train_batch_size)
    valid_loader = torch.utils.data.DataLoader(validset, shuffle=False, batch_size=args.valid_batch_size)

    evaluator = RougeEvaluator()

    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.learning_rate)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=os.path.join(args.experiment_root, 'dbpedia_'+args.model_name_or_path.split('/')[-1]+'.log'), level=logging.INFO)
    logging.info('\n')
    for k,v in vars(args).items():
        logging.info(f'{k}: {v}')

    
    for epoch in range(args.iterations):
        train(epoch, tokenizer, model, device, train_loader, optimizer, args)
        predictions, actuals = validate(epoch, tokenizer, model, device, valid_loader, args)
        
        print('-'*20+'Training Iteration: {}'.format(str(epoch))+'-'*20)

        if args.output_file:
            with open(args.output_file, 'w') as fout:
                for i, predicted in enumerate(predictions):
                    fout.write(predicted+'\n')
                    fout.write(actuals[i]+'\n')
            fout.close()
            
        results = evaluator.calculate_rouge_score(predictions, actuals)
        for k,v in results.items():
            logging.info(k+': '+str(v))

        if args.save_ckpt:
            if 'bart' in args.model_name_or_path:
                if not os.path.isdir(os.path.join(args.experiment_root, 'experiment/bart_ckpt_dbpedia')):
                    os.mkdir(os.path.join(args.experiment_root, 'experiment/bart_ckpt_dbpedia'))
                    os.mkdir(os.path.join(args.experiment_root, 'experiment/bart_ckpt_dbpedia', args.model_name_or_path.split('/')[1]+'_epoch_'+str(epoch)))
                model.save_pretrained(os.path.join(args.experiment_root, 'experiment/bart_ckpt_dbpedia', args.model_name_or_path.split('/')[1]+'_epoch_'+str(epoch)))
                tokenizer.save_pretrained(os.path.join(args.experiment_root, 'experiment/bart_ckpt_dbpedia', args.model_name_or_path.split('/')[1]+'_epoch_'+str(epoch)))
            elif 't5' in args.model_name_or_path:
                if not os.path.isdir(os.path.join(args.experiment_root, 'experiment/t5_ckpt_dbpedia')):
                    os.mkdir(os.path.join(args.experiment_root, 'experiment/t5_ckpt_dbpedia'))
                    os.mkdir(os.path.join(args.experiment_root, 'experiment/t5_ckpt_dbpedia', args.model_name_or_path+'_epoch_'+str(epoch)))
                model.save_pretrained(os.path.join(args.experiment_root, 'experiment/t5_ckpt_dbpedia', args.model_name_or_path+'_epoch_'+str(epoch)))
                tokenizer.save_pretrained(os.path.join(args.experiment_root, 'experiment/t5_ckpt_dbpedia', args.model_name_or_path+'_epoch_'+str(epoch)))