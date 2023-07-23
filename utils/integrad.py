import sys
import os
import time
import copy

import torch

from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import PreTrainedModel

from captum.attr import LayerIntegratedGradients


def generate_topk_tokens_cross_encoder(model, tokenizer, topk, input_text, n_steps=10, mode='combined'):
    assert mode in ['query-only', 'document-only', 'combined'], 'unrecognized argument, force exit!'

    act = torch.nn.Sigmoid()

    def ig_encodings(input_text):
        # adjust for bert style models
        cls_id = tokenizer.cls_token_id
        pad_id = tokenizer.pad_token_id
        sep_id = tokenizer.sep_token_id
        input_ids = tokenizer(input_text, max_length=512, truncation=True).input_ids
        base_ids = [pad_id] * len(input_ids)
        base_ids[0] = cls_id
        base_ids[-1] = sep_id

        return torch.LongTensor([input_ids]), torch.LongTensor([base_ids])

    def ig_forward(input_ids, length=1):
        outputs = model(input_ids)
        # print(outputs)
        logits = outputs[0]
        return act(logits)
    
    device = model.device

    layer = model.distilbert.embeddings
    ig = LayerIntegratedGradients(ig_forward, layer)

    input_ids, base_ids = ig_encodings(input_text)
    input_ids, base_ids = input_ids.to(device), base_ids.to(device)

    sep_id = tokenizer.sep_token_id
    input_ids_ = copy.deepcopy(input_ids)
    input_ids_list = input_ids_.squeeze().tolist()
    sep_location = input_ids_list.index(sep_id) # the first occurance of sep id
    true_class = 0
    attrs, delta = ig.attribute(
        input_ids, 
        base_ids, 
        target=true_class, 
        return_convergence_delta=True,
        n_steps=n_steps
    )

    scores = attrs.sum(dim=-1) # [seq_len]
    scores = (scores - scores.mean()) / scores.norm() # normalization
    if mode == 'query-only':
        scores = scores[:, :sep_location]
    elif mode == 'document-only':
        scores = scores[:, sep_location:]
    if scores.shape[1] <= topk:
        indices = torch.topk(scores, dim=1, k=scores.shape[1]).indices.squeeze(dim=0).cpu().tolist()
    else:
        indices = torch.topk(scores, dim=1, k=topk).indices.squeeze(dim=0).cpu().tolist()
    return [tokenizer.decode(input_ids.cpu().squeeze().tolist()[i], skip_special_tokens=True, clean_up_tokenization_spaces=True) for i in indices]


def generate_topk_tokens(model, tokenizer, topk, input_text, n_steps=10):
    def ig_encodings(input_text):
        # for T5 tokenizer there are no cls token and sep token, therefore we change the input format a bit
        pad_id = tokenizer.pad_token_id
        sep_id = tokenizer.eos_token_id
        input_ids = tokenizer(input_text).input_ids

        base_ids = [pad_id] * len(input_ids)
        input_ids = input_ids + [sep_id]
        base_ids = base_ids + [sep_id]
        return torch.LongTensor([input_ids]), torch.LongTensor([base_ids])
    
    def ig_forward(input_ids, length=1):
        decode_ids = torch.full(
            (input_ids.size(0), 1),
            model.config.decoder_start_token_id,
            dtype=torch.long
        ).to(input_ids.device)

        encoder_outputs = model.get_encoder()(input_ids, attention_mask=None)
        next_token_logits = None

        for _ in range(length):
            model_inputs = {
                "decoder_input_ids": input_ids,
                "decoder_past_key_value_states": None,
                "encoder_outputs": encoder_outputs,
                "attention_mask": None,
                "use_cache": True,
            }

            outputs = model(**model_inputs) # (batch_size, cur_len, vocab_size)
            next_token_logits = outputs[0][:, -1, :] # (batch_size, vocab_size)
            decode_ids = torch.cat([decode_ids, next_token_logits.max(1)[1].unsqueeze(-1)], dim=-1)

        # 6136: no
        # 1176: yes
        return torch.nn.functional.softmax(next_token_logits[:, [6136, 1176]], dim=1)

    device = model.device

    layer = model.shared
    ig = LayerIntegratedGradients(ig_forward, layer)

    input_ids, base_ids = ig_encodings(input_text)
    input_ids, base_ids = input_ids.to(device), base_ids.to(device)

    true_class = 1
    attrs, delta = ig.attribute(
        input_ids, 
        base_ids, 
        target=true_class, 
        return_convergence_delta=True,
        n_steps=n_steps
    )
    scores = attrs.sum(dim=-1)
    scores = (scores - scores.mean()) / scores.norm()
    indices = torch.topk(scores, dim=1, k=topk).indices.squeeze(dim=0).cpu().tolist()

    return [tokenizer.decode(input_ids.cpu().squeeze().tolist()[i], skip_special_tokens=True) for i in indices]



if __name__ == '__main__':
    model = AutoModelForSeq2SeqLM.from_pretrained('castorini/monot5-base-msmarco')
    tokenizer = AutoTokenizer.from_pretrained('t5-base')

    # model = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large-xsum')
    # tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-xsum')

    # for k,v in model.state_dict().items():
    #     print(k)
    # sys.exit()

    query = "how does solar energy compare to clean coal ?"
    document = "coal is the primary fuel for generating electricity around \
    the world . solar power can not realistically produce enough \
    energy anytime soon to replace this massive source of electricity . \
    it is incapable therefore of making a serious dent in \
    coal-electricity production and the related greenhouse \
    gas emissions"

    input_text = 'Query: '+query+' Document: '+document+' Relevant: '

    output = generate_topk_tokens(model, tokenizer, 5, input_text)
    print('output: ', output)