
# Goal: for each file, get (javadoc comment, code)
# Check which javadoc comments i want: class, method, in-method?
# Filter comments? Length of tokens?
# How to store code?
# Build tensors, convert to cuda

import os
import sys
import _pickle as pickle

from graph_pb2 import *

proto_list = "proto_list.txt"

nodeid_dict_file = "nodeids_dict.pickle"

def parse_file(file_path):
    with open(file_path, "rb") as f:
        g = Graph()
        g.ParseFromString(f.read())
        """
        for n in g.node:
            print(n.contents)
            if n.type is FeatureNode.COMMENT_JAVADOC:
                print("Javadoc comment")
        """

        """
        root = g.ast_root
        root_id = g.ast_root.id
        print("g.ast_root", root)

        for e in g.edge:
            if e.sourceId == root_id:
                print("FOUND, other is: ")
                for n in g.node:
                    if n.id == e.destinationId:
                        print(n)

            if e.destinationId == root_id:
                print("FOUND, other is: ")
                for n in g.node:
                    if n.id == e.sourceId:
                        print(n)
        """
        node_dict = {}
        for n in g.node:
            node_dict[n.id] = n

        #pickle.dump(node_dict, open(nodeid_dict_file, "wb")


        javadoc_method = 0
        javadoc_class = 0
        javadoc_variable = 0

        for e in g.edge:
            if e.type is FeatureEdge.COMMENT:
                source_node = node_dict[e.sourceId]
                destination_node = node_dict[e.destinationId]
                if source_node.type == FeatureNode.COMMENT_JAVADOC:
                    if destination_node.contents == "METHOD":
                        javadoc_method +=1
                    if destination_node.contents == "VARIABLE":
                        javadoc_variable += 1
                    if destination_node.contents == "COMPILATION_UNIT":
                        javadoc_class += 1

        return javadoc_class, javadoc_method, javadoc_variable


def build_dataset(fname= proto_list):
    f = open(fname, "r")
    content = f.readlines()
    content = [x.strip() for x in content]
    # total_token_count = 0 -> 9807850
    # total_javadoc_comment_count = 0 -> 20779
    # token_contents_list = []
    total_javadoc_class = 0
    total_javadoc_method = 0
    total_javadoc_variable = 0
    for line in content:
        # total_token_count += return_token_count(line)
        # total_javadoc_comment_count += return_javadoc_comment_count(line)
        # describe_types(line)
        # token_contents_list = get_contents(line, token_contents_list)
        javadoc_class, javadoc_met, javadoc_var = parse_file(line)
        total_javadoc_class += javadoc_class
        total_javadoc_method += javadoc_met
        total_javadoc_variable += javadoc_var

    print("Javadoc method ", total_javadoc_method)
    print("Javadoc class ", total_javadoc_class)
    print("Javadoc variable", total_javadoc_variable)


def try_parse_file():
    parse_file("/Users/andreea/Documents/Part III/MLforProg(R252)/aid25/Documentation.java.proto")

build_dataset()
