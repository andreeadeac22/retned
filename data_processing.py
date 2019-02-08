
# Goal: for each file, get (javadoc comment, code)
# Check which javadoc comments i want: class, method, in-method? -- functions of less than 150 tokens
# Filter comments? Length of tokens? ^ above
# How to store code? -- text/ vocab embedding  -- one-hot for words?
# Build tensors, convert to cuda



# Split train/test on same directory -- not possible
# White list

import os
import sys
import _pickle as pickle
import collections
import re

from graph_pb2 import *

proto_list = "proto_list.txt"

def parse_file(file_path, comment_code_dict, methodlen_dict):
    with open(file_path, "rb") as f:
        g = Graph()
        file_content = f.read()
        g.ParseFromString(file_content)

        idnode_dict = {}
        for n in g.node:
            idnode_dict[n.id] = n


        javadoc_method = 0
        javadoc_class = 0
        javadoc_variable = 0


        for e in g.edge:
            if e.type is FeatureEdge.COMMENT:
                source_node = idnode_dict[e.sourceId]
                destination_node = idnode_dict[e.destinationId]
                if source_node.type == FeatureNode.COMMENT_JAVADOC:
                    if destination_node.contents == "VARIABLE":
                        javadoc_variable += 1
                    if destination_node.contents == "COMPILATION_UNIT":
                        javadoc_class += 1

                    if destination_node.contents == "METHOD":
                        javadoc_method +=1
                        method_list = []
                        #TODO: get list of tokens
                        startpos = destination_node.startPosition
                        endpos = destination_node.endPosition
                        for n in g.node:
                            if n.startPosition >= startpos and n.endPosition <= endpos:
                                if n.type == FeatureNode.TOKEN:
                                    method_list += [n.contents]
                                if n.type == FeatureNode.IDENTIFIER_TOKEN:
                                    method_list += [n.contents]

                        method_token_length = len(method_list)

                        if method_token_length < 151:
                            if not method_token_length in methodlen_dict:
                                methodlen_dict[method_token_length] = 1
                            else:
                                methodlen_dict[method_token_length] += 1

                            if source_node.contents not in comment_code_dict:
                                comment_code_dict[source_node.contents] = method_list
                            #else:
                            #    print(source_node.contents)

        return comment_code_dict, methodlen_dict


def build_dataset(fname= proto_list):
    f = open(fname, "r")
    content = f.readlines()
    content = [x.strip() for x in content]

    comment_code_dict = {}   # key is javadoc comment, value is list of method tokens
    methodlen_dict = {}      #key is number of token per method, value is how many methods with that many tokens exist

    for line in content:
        comment_code_dict, methodlen_dict = parse_file(line, comment_code_dict, methodlen_dict)

    comment_code_file_write = open("commentcode.pickle", "wb")
    pickle.dump(comment_code_dict, comment_code_file_write)


    comment_code_txt = open("commentcode.txt", "w")
    for key in comment_code_dict:
        print(key, comment_code_dict[key], file=comment_code_txt)


    methodlen_dict_file = open("methodlen.pickle", "wb")
    pickle.dump(methodlen_dict, methodlen_dict_file)

    methodlen_dict_txt = open("methodlen.txt", "w")
    for key in methodlen_dict:
        print(key, methodlen_dict[key], file=methodlen_dict_txt)

    print("Dict length", len(comment_code_dict))


def print_tokens():
    with open("/Users/andreea/Documents/Part III/MLforProg(R252)/aid25/Documentation.java.proto", "rb") as f:
        g = Graph()
        file_content = f.read()
        g.ParseFromString(file_content)
        print("Tokens)")
        for n in g.node:
            if n.type is FeatureNode.TOKEN or n.type is FeatureNode.IDENTIFIER_TOKEN:
                print(n.contents)


def tokens2vocab():
    comment_code_file_read = open("commentcode.pickle", "rb")
    comment_code_dict = pickle.load(comment_code_file_read)

    token_freq = {}
    for comment in comment_code_dict:
        token_list = comment_code_dict[comment]
        for token in token_list:
            if token in token_freq:
                token_freq[token] += 1
            else:
                token_freq[token] = 1

    token_freq_file = open("tokenfreq.pickle", "wb")
    pickle.dump(token_freq, token_freq_file)

    token_freq_txt = open("tokenfreq.txt", "w")
    sorted_by_value = sorted(token_freq.items(), key=lambda kv: kv[1])
    for k,v in sorted_by_value:
        print(k, v, file=token_freq_txt)


def comment2list():
    comment_code_file_read = open("commentcode.pickle", "rb")
    comment_code_dict = pickle.load(comment_code_file_read)
    delimiters = ' ', "*", "\n", "/", "\t"
    regexPattern = '|'.join(map(re.escape, delimiters))

    comment2list_dict = {}
    comment2list_txt = open("comment2list.txt", "w")
    comment2list_file = open("comment2list.pickle", "wb")

    for comment in comment_code_dict:
        comment_lines = comment.splitlines()
        comment_lines = comment_lines[1:len(comment_lines)]
        word_list = []
        for comment_line in comment_lines:
            line_words = re.split(regexPattern, comment_line)
            line_words = list(filter(None, line_words))
            word_list += line_words

        comment2list_dict[comment] = word_list
        print(comment, comment2list_dict[comment], file=comment2list_txt)

    pickle.dump(comment2list_dict, comment2list_file)


def commentwords2vocab():
    comment2list_read = open("comment2list.pickle", "rb")
    comment2list_dict = pickle.load(comment2list_read)

    commentword_freq = {}
    for comment in comment2list_dict:
        word_list = comment2list_dict[comment]
        for word in word_list:
            if word in commentword_freq:
                commentword_freq[word] += 1
            else:
                commentword_freq[word] = 1

    commentword_freq_file = open("commentwordfreq.pickle", "wb")
    pickle.dump(commentword_freq, commentword_freq_file)

    commentword_freq_txt = open("commentwordfreq.txt", "w")
    sorted_by_value = sorted(commentword_freq.items(), key=lambda kv: kv[1])
    for k,v in sorted_by_value:
        print(k, v, file=commentword_freq_txt)
