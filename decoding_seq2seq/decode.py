import sys
sys.path.append('/home/sci/zhichao.xu/Constrained-QFS')

import json
import math
import os
import time
import logging
import argparse
import itertools
import numpy as np
from tqdm import tqdm
import pathlib
import copy
import csv
import re

from utils.utils import tokenize_constraints, read_constraints, expand_factor
from utils.lexical_constraints import init_batch
from utils.integrad import generate_topk_tokens_cross_encoder

from utils.input_utils import format_input, read_data_tsv
from utils.utils import get_stopwords, posthoc_filtering, get_word_forms_local

from evaluation.eval_rouge import RougeEvaluator

from finetune.finetune_dbpedia import val_targets, SummaryDataset

from decoding_seq2seq.topK import topk_huggingface, ConstrainedHypothesis
from decoding_seq2seq.generate import generate

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()

parser.add_argument("--input_dir", type=str, default='/home/sci/zhichao.xu/cg_dataset/dbpedia_processed')
parser.add_argument(
    '--input_format', 
    type=str, 
    default='t5-prompt',
    choices=['bart-concat', 'bart-context_only', 't5-summarize', 't5-prompt', 'monoT5', 'cross-encoder']
)
parser.add_argument('--output', type=str, default='./seq2seq-constrained.log')
parser.add_argument('--triples', type=str, default='test_triples.tsv')

parser.add_argument('--test_samples', type=int, default=100)
parser.add_argument('--do_save', action='store_true')

# saliency model specific
parser.add_argument('--saliency_model_name_or_path', type=str, default='castorini/monot5-base-msmarco')
parser.add_argument('--saliency_tokenizer', type=str, default='t5-base')

# decoder specific 
parser.add_argument("--model_name_or_path", type=str, default='facebook/bart-large')
parser.add_argument("--tokenizer", type=str, default='facebook/bart-large')

parser.add_argument('--constraints_mode', type=str, default='combined', choices=['query-only', 'document-only', 'combined'])

parser.add_argument('--batch_size', type=int, default=8,
                    help="Batch size for decoding.")
parser.add_argument('--beam_size', type=int, default=5,
                    help="Beam size for searching")
parser.add_argument('--max_source_length', type=int, default=150)
parser.add_argument('--max_tgt_length', type=int, default=48,
                    help="maximum length of decoded sentences")
parser.add_argument('--min_tgt_length', type=int, default=5,
                    help="minimum length of decoded sentences")
parser.add_argument('--ngram_size', type=int, default=3,
                    help='all ngrams can only occur once')
parser.add_argument('--length_penalty', type=float, default=0.1,
    help="length penalty for beam search, >0 promotes longer sequence, \
        <0 discourages longer sequence"
    )
parser.add_argument('--prune_factor', type=int, default=50,
                    help="fraction of candidates to keep based on score")
parser.add_argument('--sat_tolerance', type=int, default=2,
                    help="minimum satisfied clause of valid candidates")
parser.add_argument('--beta', type=float, default=0.1,
                    help="reward factor for in progress constraint")
parser.add_argument('--early_stop', type=float, default=None,
                    help="optional early stop if all constraints are satisfied")
parser.add_argument('--topk_constraint', type=int, default=1)

parser.add_argument(
    "--decoder_start_token_id",
    type=int,
    default=None,
    required=False,
    help="decoder_start_token_id (otherwise will look at config)",
)
parser.add_argument(
    "--n_obs", type=int, default=-1, required=False, help="How many observations. Defaults to all."
)

args = parser.parse_args()
config = vars(args)
for k,v in config.items():
    print(str(k)+': '+str(v))

stop_words = get_stopwords()

test_queries, test_contexts, test_targets = read_data_tsv(os.path.join(args.input_dir, args.triples))
test_queries = test_queries[:args.test_samples]
test_contexts = test_contexts[:args.test_samples]
test_targets = test_targets[:args.test_samples]

# construct constraints with Integrated Gradients and monoT5
integrad_model = AutoModelForSequenceClassification.from_pretrained(args.saliency_model_name_or_path)
integrad_tokenizer = AutoTokenizer.from_pretrained(args.saliency_tokenizer)

# from here we start to build constraints
integrad_model = integrad_model.cuda() if torch.cuda.is_available() else integrad_model.cpu()

input_lines_w_constraint, constraints_list, test_targets_w_constraint = [], [], []
input_lines_wo_constraint, test_targets_wo_constraint = [], []
test_contexts_w_constraint, test_contexts_wo_constraint = [], []
for i, row in tqdm(enumerate(test_queries), desc='creating saliency map...'):
    query, context, target = test_queries[i], test_contexts[i], test_targets[i]
    
    constraints = generate_topk_tokens_cross_encoder(
        model=integrad_model, 
        tokenizer=integrad_tokenizer, 
        topk=10, 
        input_text=format_input(query, context, 'cross-encoder'), 
        n_steps=10,
        mode=args.constraints_mode
        )
    constraints = ["".join(i.split()) for i in constraints]
    constraints = [[i] for i in posthoc_filtering(stop_words, constraints) if i[:2] != "##"]
    # print(constraints)
    expanded_constraints = get_word_forms_local(constraints)
    if len(expanded_constraints) == 0:
        input_lines_wo_constraint.append(format_input(query, context, args.input_format))
        test_contexts_wo_constraint.append(context)
        test_targets_wo_constraint.append(target)
    else:
        constraints_list.append(expanded_constraints)
        input_lines_w_constraint.append(format_input(query, context, args.input_format))
        test_contexts_w_constraint.append(context)
        test_targets_w_constraint.append(target)

del integrad_model, integrad_tokenizer

assert len(input_lines_w_constraint) == len(constraints_list), 'length mismatch, forced exit!'
assert len(input_lines_w_constraint) == len(test_targets_w_constraint), 'length mismatch, forced exit!'
print('Num of total test instances w. constraint: {}'.format(len(input_lines_w_constraint)))

# from here we official decode
print(f"Decoding with: {args.model_name_or_path}")
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, model_max_length=512, truncation=True)
model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)

valid_inputs = [input_lines_w_constraint, test_targets_w_constraint]
validset = SummaryDataset(valid_inputs, tokenizer, args.max_source_length, args.max_tgt_length)
valid_loader = torch.utils.data.DataLoader(validset, shuffle=False, batch_size=args.batch_size)
actuals = val_targets(tokenizer, valid_loader)

valid_inputs_ = [input_lines_wo_constraint, test_targets_wo_constraint]
validset_ = SummaryDataset(valid_inputs_, tokenizer, args.max_source_length, args.max_tgt_length)
valid_loader_ = torch.utils.data.DataLoader(validset_, shuffle=False, batch_size=args.batch_size)
actuals_ = val_targets(tokenizer, valid_loader_)

assert len(actuals) == len(input_lines_w_constraint), 'length mismatch, forced exit!'
assert len(actuals_) == len(input_lines_wo_constraint), 'length mismatch, forced exit!'

model.eval()
if torch.cuda.is_available():
    model = model.to('cuda')
    torch.cuda.empty_cache()

summarize_output_ = []
with torch.no_grad():
    for i, data in tqdm(enumerate(valid_loader_), desc='decoding non-constraints instances...'):
        y = data['target_ids'].to(DEFAULT_DEVICE, dtype = torch.long)
        ids = data['source_ids'].to(DEFAULT_DEVICE, dtype = torch.long)
        mask = data['source_mask'].to(DEFAULT_DEVICE, dtype = torch.long)
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
        preds = tokenizer.batch_decode(generated_ids.cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
        summarize_output_.extend(preds)

period_id = [tokenizer.convert_tokens_to_ids('.')]
period_id.append(tokenizer.convert_tokens_to_ids('Ġ.'))
eos_ids = [tokenizer.eos_token_id] + period_id
PAD_ID = tokenizer.convert_tokens_to_ids('<pad>')
bad_token = [':', "'", '-', '_', '@', 'Ċ', 'Ġ:', 'Ġwho', "'s"]
bad_words_ids = [tokenizer.convert_tokens_to_ids([t]) for t in bad_token]

init_factor = [1 for x in input_lines_w_constraint]
input_lines_w_constraint = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x)) for x in input_lines_w_constraint]

constraints_list = tokenize_constraints(tokenizer, constraints_list)
constraints_list = expand_factor(constraints_list, init_factor)

assert len(input_lines_w_constraint)==len(constraints_list), 'length mismatch, forced exit!'
print('finished decoding the non-constraints instances')

total_batch = math.ceil(len(input_lines_w_constraint) / args.batch_size)
next_i = 0

summarize_output = []

pbar=tqdm(total=total_batch)
while next_i < len(input_lines_w_constraint):
    _chunk = input_lines_w_constraint[next_i:next_i + args.batch_size]
    constraints = init_batch(raw_constraints=constraints_list[next_i:next_i + args.batch_size],
                                beam_size=args.beam_size,
                                eos_id=eos_ids)
    buf = _chunk
    next_i += args.batch_size

    max_len = max([len(x) for x in buf])
    buf = [x + [PAD_ID] * (max_len - len(x)) for x in buf]

    input_ids = torch.stack([torch.from_numpy(np.array(x)) for x in buf])
    input_ids = input_ids.to('cuda')
    attention_mask = (~torch.eq(input_ids, PAD_ID)).int()
    attention_mask = attention_mask.to('cuda')

    advanced_constraints = []
    for j, init_cons in enumerate(constraints):
        adv_cons = init_cons
        for token in _chunk[j // args.beam_size]:
            adv_cons = adv_cons.advance(token, False)
        advanced_constraints.append(adv_cons)

    # for seq2seq lm, we do not adopt advanced constraints
    summaries = generate(self=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_start_token_id=tokenizer.bos_token_id,
                    min_length=args.min_tgt_length,
                    max_length=args.max_tgt_length,
                    num_beams=args.beam_size,
                    no_repeat_ngram_size=args.ngram_size,
                    length_penalty=args.length_penalty,
                    constraints=constraints,
                    prune_factor=args.prune_factor,
                    sat_tolerance=args.sat_tolerance,
                    beta=args.beta,
                    early_stop=args.early_stop
                    )
    # print(summaries)
    summaries = tokenizer.batch_decode(summaries.cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
    # summaries = tokenizer.batch_decode(summaries.cpu())
    summarize_output.extend(summaries)
    pbar.update(1)
pbar.close()

summarize_output.extend(summarize_output_)
actuals.extend(actuals_)
test_contexts_w_constraint.extend(test_contexts_wo_constraint)
assert len(summarize_output)==len(actuals), 'length mismatch, forced exit!'
assert len(summarize_output)==len(test_contexts_w_constraint), 'length mismatch, forced exit!'

logging.basicConfig(filename='./decode-cross-encoder.log', level=logging.INFO)
logging.info('\n')
for k,v in vars(args).items():
    logging.info(f'{k}: {v}')

evaluator = RougeEvaluator()
result_dict = evaluator.calculate_rouge_score(summarize_output, actuals)
test_targets_w_constraint.extend(test_contexts_wo_constraint)
result_dict = evaluator.calculate_rouge_score(summarize_output, test_targets_w_constraint)
for k,v in result_dict.items():
    logging.info(f'{k}: {v}')

with open(args.output, 'w') as fout:
    tsv_writer = csv.writer(fout, delimiter='\t')
    for i, row in enumerate(summarize_output):
        tsv_writer.writerow([summarize_output[i], test_contexts_w_constraint[i], actuals[i]])
fout.close()