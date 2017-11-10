# -*- coding: utf-8 -*-
from data_helpers import read_data, build_dataset, batch_iter, get_words
from Model import WordCharModel
import tensorflow as tf
import numpy as np
import os
import time
import gc

filename = '../data/wiki_cut.txt'
batch_size = 64
num_epochs = 20
window_size = 7
min_time = 5
embedding_size = 100
num_sampled = 16
start_lr = 1.5 / batch_size

datas, words = read_data(filename)
dataset, word_dictionary, character_dict, wordID_charID, total_instance, dictionary_pro = build_dataset(datas, words)
reverse_dictionary = dict(zip(word_dictionary.values(), word_dictionary.keys()))

del datas, words
gc.collect()

valid_size = 6
valid_examples = np.array([word_dictionary['街道'], word_dictionary['教授'], word_dictionary['医生'],
                           word_dictionary['英里'], word_dictionary['计算机'], word_dictionary['老虎']])

model = WordCharModel(word_size=len(word_dictionary),
                      character_size=len(character_dict),
                      embedding_size=embedding_size,
                      num_sampled=num_sampled,
                      valid_examples=valid_examples,
                      dictionary_pro=dictionary_pro)

loss = model.loss
train = model.train
save_embed = model.save_embed
similarity = model.similarity

init = tf.global_variables_initializer()

session = tf.Session()
session.run(init)
print("Initialized")

average_loss = 0

lr = start_lr
word_count_actual = 0
total_step = total_instance // batch_size

for epoch in range(num_epochs):
    step = 0
    batchs = batch_iter(batch_size=batch_size, dataset=dataset, wordID_charID=wordID_charID, window_size=window_size)
    for batch in batchs:
        word_count_actual += batch_size

        lr = start_lr*(1-word_count_actual/(num_epochs*total_instance+1.0))
        if lr < start_lr*0.01:
            lr = start_lr*0.01

        old_time = time.time()
        step += 1
        words, charaters, seq_len, window_size_now, input_y = batch

        feed_dict = {model.input_word: words,
                     model.input_character: charaters,
                     model.seq_len: seq_len,
                     model.window_size: window_size_now,
                     model.input_y: input_y,
                     model.lr: lr}
        _, loss_val = session.run([train, loss], feed_dict=feed_dict)
        average_loss += np.average(loss_val)
        if step % 20000 == 0:
            cost_time = time.time() - old_time
            wps = int(batch_size / cost_time)
            if step > 0:
                average_loss /= 20000
            # The average loss is an estimate of the loss over the last 20000 batches.
            print("epoch %d step %d/%d Average loss: %f. lr: %f. Each batch takes %f sec, %d wps" % (
                epoch, step, total_step, average_loss, lr, cost_time, wps))
            
        if step % (total_step // 3) == 0:
            sim = session.run(similarity)
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = "Nearest to %s:" % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = "%s %s," % (log_str, close_word)
                print(log_str)
            print('------------------------------------------------')
            print


print('Save embedding')
word_batchs = get_words(word_dictionary, wordID_charID, batch_size, filename='words(word_model).txt')
writer = open('embed(word_model).txt', 'w')
for batch in word_batchs:
    word_save, char_save, seq_len_save, window_size_save = batch
    feed_dict = {model.input_word: word_save,
                 model.input_character: char_save,
                 model.seq_len: seq_len_save,
                 model.window_size: window_size_save}
    word_embed_value = session.run([save_embed], feed_dict=feed_dict)

    for value in word_embed_value:
        np.savetxt(writer, value, fmt='%.6f')
writer.close()

words_reader = open('words(word_model).txt', 'r')
embed_reader = open('embed(word_model).txt', 'r')

words = words_reader.readlines()
embeds = embed_reader.readlines()

result = open('result(word_model_CBOW).txt', 'w')
result.write(str(len(words)) + ' ' + str(embedding_size) + '\n')
for word, embed in zip(words, embeds):
    word = word.strip()
    embed = embed.strip()
    line = word + ' ' + embed + '\n'
    num = len(line.split(' '))
    result.write(word + ' ' + embed + '\n')
result.close()

os.remove('words(word_model).txt')
os.remove('embed(word_model).txt')
