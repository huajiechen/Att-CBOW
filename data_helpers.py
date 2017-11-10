# -*- coding: utf-8 -*-
import collections
import time

import random
import numpy as np


# Read the data into a list of strings.
def read_data(filename):
    print('load corpus.')
    old_time = time.time()
    f = open(filename, 'r')
    words = []
    datas = []
    for line in f.readlines():
        line = line.strip()+' </s>'
        line = line.split()
        words.extend(line)
        datas.append(line)
    print('finish loading corpus. There are %d words. Cost %d sec' % (len(words), time.time() - old_time))
    return datas, words


def build_dataset(datas, words, window_size=2, min_time=5):
    print('Creat word dictionary.')
    old_time = time.time()
    word_dictionary = dict()
    character_dict = {' ': 0}
    wordID_charID = dict()
    # dictionary['UNK'] = 0
    count = collections.Counter(words).most_common()
    dictionary_pro = []
    for word, freq in count:
        if freq >= min_time:
            word_dictionary[word] = len(word_dictionary)
            dictionary_pro.append(pow(freq, 0.75))
            charID_list = []
            wordID = word_dictionary[word]
            word = word.decode('utf-8')
            for char in word:
                if char not in character_dict:
                    character_dict[char] = len(character_dict)
                charID_list.append(character_dict[char])
            charID_list = charID_list[:6]+[0]*max(0, 6-len(charID_list))
            wordID_charID[wordID] = charID_list

    dataset = list()

    num = 0
    for line in datas:
        if len(line) < 2*window_size+1:
            continue
        sentence = []
        for word in line:
            if word in word_dictionary:
                index = word_dictionary[word]
                sentence.append(index)
            else:
                continue
        dataset.append(sentence)
        num += len(sentence)-2*window_size
    print('finish creating word dictionary. There are %d words. Cost %d sec' % (len(word_dictionary), time.time() - old_time))
    # print('shuffle data')
    # random.shuffle(dataset)
    # print('finish shuffling data')
    return dataset, word_dictionary, character_dict, wordID_charID, num, dictionary_pro


def batch_iter(batch_size, dataset, wordID_charID, window_size=2):
    x = []
    y = []
    char = []
    seq_len = []
    b = random.randint(0, window_size-1)
    for line_index in range(len(dataset)):
        for word_index in range(window_size, len(dataset[line_index])-window_size):
            context = []
            context_char = []
            context_char_len = []
            for k in range(-window_size+b, window_size + 1-b):
                if k == 0:
                    y.append([dataset[line_index][word_index]])
                else:
                    context.append(dataset[line_index][word_index + k])
                    charID = wordID_charID[dataset[line_index][word_index + k]]
                    context_char.append(charID)
                    char_len = 0
                    for id in charID:
                        if id != 0:
                            char_len += 1
                    context_char_len.append(char_len)
            x.append(context)
            char.append(context_char)
            seq_len.append(context_char_len)
            if len(y) >= batch_size:
                data_words = x
                data_y = y
                data_seq_len = seq_len
                data_char = char
                window_size_now = (window_size-b)*2

                x = []
                y = []
                char = []
                seq_len = []
                b = random.randint(0, window_size - 1)
                yield np.array(data_words, dtype=np.int32), \
                      np.array(data_char, dtype=np.int32), \
                      np.array(data_seq_len, dtype=np.int32), \
                      np.array(window_size_now, dtype=np.int32),\
                      np.array(data_y, dtype=np.int32)

    if len(x) > 0:
        data_words = x
        data_y = y
        data_seq_len = seq_len
        data_char = char
        window_size_now = (window_size-b)*2

        yield np.array(data_words, dtype=np.int32), \
              np.array(data_char, dtype=np.int32), \
              np.array(data_seq_len, dtype=np.int32), \
              np.array(window_size_now, dtype=np.int32), \
              np.array(data_y, dtype=np.int32)


def get_words(dictionary_word, wordID_charID, batch_size, filename):
    writer = open(filename, 'w')
    words= dictionary_word.keys()
    x = []
    char = []
    seq_len = []
    for word in words:
        writer.write(word+'\n')
        writer.flush()
        charID = wordID_charID[dictionary_word[word]]
        char_len = 0
        for id in charID:
            if id != 0:
                char_len += 1

        x.append([dictionary_word[word]])
        char.append([charID])
        seq_len.append([char_len])

        if len(x) >= batch_size:
            data_word = x
            data_char = char
            data_seq_len = seq_len
            x = []
            char = []
            seq_len = []
            yield np.array(data_word, dtype=np.int32), \
                  np.array(data_char, dtype=np.int32), \
                  np.array(data_seq_len, dtype=np.int32),\
                  np.array(1, dtype=np.int32)
    if len(x) > 0:
        data_word = x
        data_char = char
        data_seq_len = seq_len
        yield np.array(data_word, dtype=np.int32),\
              np.array(data_char, dtype=np.int32),\
              np.array(data_seq_len, dtype=np.int32),\
              np.array(1, dtype=np.int32)


if __name__ == '__main__':
    filename = '../data/test.txt'
    print('二手车')
    datas, words = read_data(filename)
    dataset, word_dictionary, character_dict, wordID_charID, num = build_dataset(datas, words, min_time=0)

    wordID_word = dict(zip(word_dictionary.values(), word_dictionary.keys()))
    charID_char = dict(zip(character_dict.values(), character_dict.keys()))
    word_id = word_dictionary['二手车']
    print(wordID_word[word_id])
    for id in wordID_charID[word_id]:
        print(charID_char[id])




