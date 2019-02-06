
# Goal: for each file, get (javadoc comment, code)
# Check which javadoc comments i want: class, method, in-method? -- functions of less than 150 tokens
# Filter comments? Length of tokens? ^ above
# How to store code? -- text/ vocab embedding  -- one-hot for words?
# Build tensors, convert to cuda

import os
import sys
import _pickle as pickle

from graph_pb2 import *

proto_list = "proto_list.txt"

nodeid_dict_file = "nodeids_dict.pickle"

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

    comment_code_file = open("commentcode.pickle", "wb")
    pickle.dump(comment_code_dict, comment_code_file)

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
#get_text_chunk()
#print_tokens()
#try_parse_file()
#build_dataset()
