import torch
from typing import Union, Tuple

from transformers import PreTrainedModel

DecodedOutput = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]

def greedy_decode(model: PreTrainedModel,
                  input_ids: torch.Tensor,
                  length: int,
                  attention_mask: torch.Tensor = None,
                  return_last_logits: bool = True) -> DecodedOutput:
    decode_ids = torch.full((input_ids.size(0), 1),
                            model.config.decoder_start_token_id,
                            dtype=torch.long).to(input_ids.device)
    encoder_outputs = model.get_encoder()(input_ids, attention_mask=attention_mask)
    next_token_logits = None
    # TODO
    # this line is not compatible in transformers==3.0.2
    for _ in range(length):
        model_inputs = model.prepare_inputs_for_generation(
            decode_ids,
            encoder_outputs=encoder_outputs,
            past=None,
            attention_mask=attention_mask,
            use_cache=True
        )

        outputs = model(**model_inputs)  # (batch_size, cur_len, vocab_size)
        next_token_logits = outputs[0][:, -1, :]  # (batch_size, vocab_size)
        decode_ids = torch.cat([decode_ids,
                                next_token_logits.max(1)[1].unsqueeze(-1)],
                               dim=-1)
    if return_last_logits:
        return decode_ids, next_token_logits
    return decode_ids

class RelevanceModel(torch.nn.Module):
    def __init__(self):
        super(RelevanceModel, self).__init__()
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        self.model = T5ForConditionalGeneration.from_pretrained('castorini/monot5-base-msmarco')
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
    
    def forward(self, query, document):
        input_text = 'Query: '+query+' Document: '+document+' Relevant: '
        input_ids = self.tokenizer(input_text, return_tensors='pt').input_ids

        _, batch_scores = greedy_decode(self.model, input_ids, length=1, return_last_logits=True)
        batch_scores = batch_scores[:, [6136, 1176]] # ['False', 'True']
        batch_scores = torch.nn.functional.softmax(batch_scores, dim=1)
        
        return batch_scores.squeeze(dim=0)

def gradient_calculator(gradient_tensor, embedding_tensor):
    # use the same gradient attribution method as https://arxiv.org/pdf/2010.05419.pdf and https://arxiv.org/pdf/1804.07781.pdf
    assert gradient_tensor.shape[0] == 1, 'wrong gradient tensor shape, force exit'
    assert embedding_tensor.shape[0] == 1, 'wrong embedding tensor shape, force exit'
    assert gradient_tensor.shape == embedding_tensor.shape, 'mismatch tensor shape, force exit'

    tensor_mul = torch.mul(gradient_tensor, embedding_tensor).squeeze(dim=0)
    # print(torch.norm(tensor_mul, dim=1)/torch.norm(tensor_mul))
    token_contrib = torch.norm(tensor_mul, dim=1)/torch.norm(tensor_mul)
    return token_contrib