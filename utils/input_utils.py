import sys
import os
import time

def format_input(query, context, input_format):
    if input_format == 'bart-concat':
        return f"{context} </s> {query}"
    elif input_format == 'bart-context_only':
        return f"Summarize: {context}"
    elif input_format == 't5-summarize':
        return f"Summarize: {context} </s> {query}"
    elif input_format == 'cross-encoder':
        return f"{query} [SEP] {context}"

def read_data_tsv(input_file):
    queries, sources, targets = [], [], []
    with open(input_file, 'r') as fin:
        for line in fin:
            query, source, target = line.strip().split('\t')
            queries.append(query)
            sources.append(source)
            targets.append(target)
    fin.close()
    return queries, sources, targets