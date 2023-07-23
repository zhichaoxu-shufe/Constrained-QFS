import sys
import os
import time
from word_forms.word_forms import get_word_forms

def tokenize_constraints(tokenizer, raw_cts):
    def tokenize(phrase):
        tokens = tokenizer.tokenize(phrase)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        return token_ids, True
    return [[list(map(tokenize, clause)) for clause in ct] for ct in raw_cts]

def read_constraints(file_name):
    cons_list = []
    with open(file_name, 'r') as f:
        for i, line in enumerate(f):
            cons = []
            for concept in json.loads(line):
                cons.append([f' {c}' for c in concept if c.islower()])
            cons_list.append(cons)
    return cons_list

def expand_factor(items, factors):
    expanded_items = []
    for item, factor in zip(items, factors):
        expanded_items.extend([item] * factor)
    return expanded_items

def str2bool(v):
	return str(v).lower() in ("yes", "true", "t", "1")

def get_stopwords():
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    to_add = [
        "Query", "Document", "Relevan"
        "1", "2", "3", "4", "5", 
        "6", "7", "8", "9", "0", 
        "", "ly", "<unk>", "th", "ok"
        "via", "a", "b", "c", "d",
        "e", "f", "g", "h", "i", "j",
        "k", "l", "m", "n", "o", "p",
        "q", "r", "s", "t", "u", "v",
        "w", "x", "y", "z", ""
        "11", "12", "13", "14", "15",
        "16", "17", "18", "19", "20",
        "21", "22", "23", "24", "25",
        "26", "27", "28", "29", "30",
        "31", "32", "33", "34", "35",
        "36", "37", "38", "39", "40",
        "41", "42", "43", "44", "45",
        "46", "47", "48", "49", "50",
        "51", "52", "53", "54", "55",
        "56", "57", "58", "59", "60",
        "61", "62", "63", "64", "65",
        "66", "67", "68", "69", "80",
        "71", "72", "73", "74", "75",
        "76", "77", "78", "79", "80",
        "81", "82", "83", "84", "85",
        "86", "87", "88", "89", "90",
        "91", "92", "93", "94", "95",
        "96", "97", "98", "99", 
        "[CLS]", "[SEP]", "[UNK]", "</s>",
        "<pad>", "</s>", "...",
    ]
    import string
    punkt = list(string.punctuation)
    return stop_words+punkt+to_add
    
def posthoc_filtering(stopwords, integrad_output):
    return list(set([i for i in integrad_output if i not in stopwords]))

def get_word_forms_local(constraints):
    constraints_out = []
    for i in constraints:
        many_word_forms = get_word_forms(i[0])
        merged = []
        for k,v in many_word_forms.items():
            merged += list(v)
        if len(merged) == 0:
            continue
        else:
            constraints_out.append(list(set(merged)))
        
    return constraints_out

if __name__ == '__main__':
    constraints = [['demand'], ['industry'], ['health'], ['sector'], ['vaccine']]
    print(get_word_forms_local(get_word_forms_local(constraints)))