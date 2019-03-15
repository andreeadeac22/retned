import csv
import pickle

from data_processing import *

def hstone_build_dataset(hstone_dataset_name):
    comment_code_dict = {}
    max_code_len = 0

    all_data = []
    with open("hstone_train.tsv") as tsvfile1:
        reader1 = csv.reader(tsvfile1, delimiter='\t')
        with open("hstone_valid.tsv") as tsvfile2:
          reader2 = csv.reader(tsvfile2, delimiter='\t')
          with open("hstone_test.tsv") as tsvfile3:
            reader3 = csv.reader(tsvfile3, delimiter='\t')
            for row in reader1:
                all_data += [row]
            for row in reader2:
                all_data += [row]
            for row in reader3:
                all_data += [row]

    for id, row in enumerate(all_data):
        num_components = len(row)
        comment_list = []
        for j in range(num_components-1):
            comment_list += [row[j]]
        comment = " ".join(comment_list)

        code = row[-1].split(' ')
        code = list(filter(lambda x: not x.isdigit(), code))

        if len(code) > max_code_len:
            max_code_len = len(code)

        comment_code_dict[comment] = code

    print("max_code_len " , max_code_len)
    with open(hstone_dataset_name + "commentcode.pickle", "wb") as comment_code_file_write:
        pickle.dump(comment_code_dict, comment_code_file_write)

    with open(hstone_dataset_name + "commentcode.txt", "w") as comment_code_txt:
        for key in comment_code_dict:
            print(key, comment_code_dict[key], file=comment_code_txt)

    tokenfreq(hstone_dataset_name)
    hstone_comment2list(hstone_dataset_name)
    build_intmatrix(hstone_dataset_name)
    build_vocabs(frequency_threshold = 0, dataset_name=hstone_dataset_name)
    build_inverse_comment_dict(hstone_dataset_name)

def hstone_comment2list(dataset_name):
    with open(dataset_name + "commentcode.pickle", "rb") as comment_code_file_read:
        comment_code_dict = pickle.load(comment_code_file_read)
    delimiters = ' ', "*", "\n", "/", "\t", "(", ")", "<", ">", "{", "}"
    regexPattern = '|'.join(map(re.escape, delimiters))

    comment2list_dict = {}
    comment2list_txt = open(dataset_name + "comment2list.txt", "w")

    for comment in comment_code_dict:
        word_list = re.split(regexPattern, comment)
        comment2list_dict[comment] = word_list
        print(comment, comment2list_dict[comment], file=comment2list_txt)

    with open(dataset_name + "comment2list.pickle", "wb") as comment2list_file:
        pickle.dump(comment2list_dict, comment2list_file)

#hstone_build_dataset("hstone")
