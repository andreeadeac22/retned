import torch
import _pickle as pickle

from constants import *

def collapse_list2string(l, word2idcomment_dict):
    res = ""
    for el in l:
        if el in word2idcomment_dict:
            res += str(el) + " "
        else:
            res += "UNK "
    return res

def tensor2wordlist(t):
    commentvocab_dict = pickle.load(open("commentvocab.pickle", "rb"))
    wordlist = []
    for i in range(t.shape[0]):
        wordid = t[i].item()
        if wordid in commentvocab_dict:
            #print("wordid ", wordid)
            #print("word ", commentvocab_dict[wordid])
            wordlist += [commentvocab_dict[wordid]]
        if wordid == src_vocab_size -1 :
            #print("wordid ", wordid)
            #print("word ", "UNK")
            wordlist += ["UNK"]

    return wordlist

"""
ex_tensor = torch.Tensor([ 2627,   885,  2841,  6070,  2074,  8113,  2823,  3449,     0, 0,     0,     0,     0,     0,     0,     0,     0,     0, 0,     0,     0,     0,     0,     0,     0,     0,     0, 0,     0,     0,     0,     0,     0,     0,     0,     0])
ex_wordlist = tensor2wordlist(ex_tensor)

collapsed = collapse_list2string(ex_wordlist, pickle.load(open("word2idcommentvocab.pickle", "rb")))

wordlist2comment_dict = pickle.load(open("wordlist2comment.pickle", "rb"))

print(wordlist2comment_dict[collapsed])
"""
