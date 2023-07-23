import sys
sys.path.append('/home/sci/zhichao.xu/Constrained-QFS/')

import os
import time
import argparse
import logging
import random
random.seed(42)

from tqdm import tqdm
import numpy as np

import torch
torch.manual_seed(42)
from torch.utils.tensorboard import SummaryWriter

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


def train(epoch, writer, tokenizer, model, device, loader, optimizer, args):
    model.train()
    avg_epoch_loss = 0

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    epoch_steps = len(loader)
    for i, data in tqdm(enumerate(loader), desc='training...'):
        y = data['target_ids'].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype=torch.long)
        mask = data['source_mask'].to(device, dtype=torch.long)
    
        with torch.cuda.amp.autocast(enabled=args.fp16):
            outputs = model(input_ids=ids, attention_mask=mask, decoder_input_ids=y_ids, labels=lm_labels)
            loss = outputs[0]
        avg_epoch_loss += loss.item()
        writer.add_scalar('loss per step', loss.item(), epoch*epoch_steps+i)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    print('Epoch loss: {}'.format(avg_epoch_loss/len(loader)))

def train_multi_epoch(tokenizer, model, device, loader, optimizer, args):
    writer = SummaryWriter()

    avg_loss = 0
    total_steps = len(loader)*args.training_iterations
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    model.train()
    counter = 0
    for batch_idx, batch in enumerate(loader):
        counter += 1
        y = batch['target_ids'].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = batch['source_ids'].to(device, dtype=torch.long)
        mask = batch['source_mask'].to(device, dtype=torch.long)
        with torch.cuda.amp.autocast(enabled=args.fp16):
            outputs = model(input_ids=ids, attention_mask=mask, decoder_input_ids=y_ids, labels=lm_labels)
            loss = outputs[0]
        avg_loss += loss.item()

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        writer.add_scalar('avg loss', avg_loss/counter)


def validate(epoch, tokenizer, model, device, loader, args):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(loader), desc='evaluating...'):
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

            # if (i+1) % 100 == 0:
            #     print('{} | {} total steps done!'.format((i+1), len(loader)))

            predictions.extend(preds)
            actuals.extend(target)
    return predictions, actuals

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_dir', type=str, default='/home/sci/zhichao.xu/cg_dataset/pubmedqa_processed')
    parser.add_argument('--experiment_root', type=str, required=True)
    parser.add_argument('--output_file', type=str)

    parser.add_argument('--tokenizer', type=str, default='t5-base')
    parser.add_argument('--model_name_or_path', type=str, default='t5-base')

    parser.add_argument('--source_max_length', type=int, default=512)
    # parser.add_argument('--pad_to_max_length', type=str2bool, default='true')
    parser.add_argument('--max_tgt_length', type=int, default=96)
    parser.add_argument('--min_tgt_length', type=int, default=48)

    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument(
        '--input_format', 
        type=str, 
        default='t5-prompt',
        choices=['bart-concat', 'bart-context_only', 't5-summarize', 't5-prompt']
    )
    parser.add_argument('--valid_batch_size', type=int, default=4)

    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--learning_rate', type=float, default=3e-5)

    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--training_iterations', type=int, default=10)
    parser.add_argument('--save_ckpt', action='store_true')

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')

    args = parser.parse_args()

    for k,v in vars(args).items():
        print(str(k) + ': ' + str(v))

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)

    if 'bart' in args.model_name_or_path:
        p_prefix = '<P>'
        q_prefix = '<Q>'
        new_tokens = [p_prefix, q_prefix]
        tokenizer.add_tokens(list(new_tokens))
        model.resize_token_embeddings(len(tokenizer))

    train_queries, train_contexts, train_targets = read_data_tsv(os.path.join(args.input_dir, 'train_triples.tsv'))
    valid_queries, valid_contexts, valid_targets = read_data_tsv(os.path.join(args.input_dir, 'test_triples_truncated.tsv'))

    train_inputs = []
    for i, query in enumerate(train_queries):
        if 'bart' in args.model_name_or_path:
            train_inputs.append(format_input(q_prefix+query, p_prefix+train_contexts[i], args.input_format))
        else:
            train_inputs.append(format_input(query, train_contexts[i], args.input_format))
    train_inputs = [train_inputs, train_targets]

    valid_inputs = []
    for i, query in enumerate(valid_queries):
        if 'bart' in args.model_name_or_path:
            valid_inputs.append(format_input(q_prefix+query, p_prefix+valid_contexts[i], args.input_format))
        else:
            valid_inputs.append(format_input(query, valid_contexts[i], args.input_format))
    valid_inputs = [valid_inputs, valid_targets]

    trainset = SummaryDataset(train_inputs, tokenizer, args.source_max_length, args.max_tgt_length)
    validset = SummaryDataset(valid_inputs, tokenizer, args.source_max_length, args.max_tgt_length)
    train_loader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=args.train_batch_size)
    valid_loader = torch.utils.data.DataLoader(validset, shuffle=False, batch_size=args.valid_batch_size)

    counter = 0
    for i, batch in enumerate(valid_loader):
        counter += batch['target_ids'].shape[1]

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    evaluator = RougeEvaluator()

    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.learning_rate)
    # train_multi_epoch(tokenizer, model, device, train_loader, optimizer, args)

    writer = SummaryWriter()
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=os.path.join(args.experiment_root, 'pubmedqa_'+args.model_name_or_path.split('/')[-1]+'.log'), level=logging.INFO)
    logging.info('\n')
    for k,v in vars(args).items():
        logging.info(f'{k}: {v}')

    logging.info('train start')
    for epoch in range(args.training_iterations):
        if args.do_train:
            train(epoch, writer, tokenizer, model, device, train_loader, optimizer, args)
        if args.do_eval:
            predictions, actuals = validate(epoch, tokenizer, model, device, valid_loader, args)
        
            results = evaluator.calculate_rouge_score(predictions, actuals)
            for k,v in results.items():
                logging.info(k+': '+str(v))

            print('-'*20+'Training Iteration: {}'.format(str(epoch))+'-'*20)
        if args.save_ckpt:
            if not os.path.isdir(os.path.join(args.experiment_root, 'experiment')):
                os.mkdir(os.path.join(args.experiment_root, 'experiment'))
            if 'bart' in args.model_name_or_path:
                if not os.path.isdir(os.path.join(args.experiment_root, 'experiment/bart/pubmedqa_checkpoints')):
                    os.mkdir(os.path.join(args.experiment_root, 'experiment/bart/pubmedqa_checkpoints'))
                    os.mkdir(os.path.join(args.experiment_root, 'experiment/bart/pubmedqa_checkpoints', args.model_name_or_path.split('/')[1]+'_epoch_'+str(epoch)))
                model.save_pretrained(os.path.join(args.experiment_root, 'experiment/bart/pubmedqa_checkpoints', args.model_name_or_path.split('/')[1]+'_epoch_'+str(epoch)))
                tokenizer.save_pretrained(os.path.join(args.experiment_root, 'experiment/bart/pubmedqa_checkpoints', args.model_name_or_path.split('/')[1]+'_epoch_'+str(epoch)))
            elif 't5' in args.model_name_or_path:
                if not os.path.isdir(os.path.join(args.experiment_root, 'experiment/t5/pubmedqa_checkpoints')):
                    os.mkdir(os.path.join(args.experiment_root, 'experiment/t5/pubmedqa_checkpoints'))
                    os.mkdir(os.path.join(args.experiment_root, 'experiment/t5/pubmedqa_checkpoints', args.model_name_or_path+'_epoch_'+str(epoch)))
                model.save_pretrained(os.path.join(args.experiment_root, 'experiment/t5/pubmedqa_checkpoints', args.model_name_or_path+'_epoch_'+str(epoch)))
                tokenizer.save_pretrained(os.path.join(args.experiment_root, 'experiment/t5/pubmedqa_checkpoints', args.model_name_or_path+'_epoch_'+str(epoch)))