import tensorflow as tf
from tensorflow.python.ops import candidate_sampling_ops
import tensorflow.contrib.layers as layers

L2_REG = 1e-4


class WordCharModel(object):
    def __init__(self, word_size, character_size, embedding_size, num_sampled, valid_examples=None, dictionary_pro=None):
        self.embedding_size = embedding_size
        self.input_word = tf.placeholder(tf.int64, [None, None], name="input_word")
        self.input_character = tf.placeholder(tf.int32, [None, None, 6], name="input_character")
        self.seq_len = tf.placeholder(tf.int32, [None, None], name='seq_len')
        self.input_y = tf.placeholder(tf.int64, [None, 1], name="input_y")
        self.lr = tf.placeholder(tf.float64, name="input_y")
        self.window_size = tf.placeholder(tf.int32, name="window_size")

        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        # calculate embedding from word
        with tf.variable_scope('word_encoder'):
            init = 0.5 / embedding_size
            word_embeddings = tf.get_variable(name='embedding',
                                              initializer=tf.random_uniform([word_size, embedding_size], -init, init))
            word_embed = tf.nn.embedding_lookup(word_embeddings, self.input_word)
            word_embed = tf.reshape(word_embed, [-1, embedding_size])

        # calculate embedding from character
        with tf.variable_scope('character_encoder'):
            init = 0.5 / embedding_size
            character_embeddings = tf.concat(concat_dim=0, values=[tf.zeros([1, embedding_size]),
                                                                   tf.random_uniform([character_size - 1, embedding_size], -init,init)])
            char_embed = tf.nn.embedding_lookup(character_embeddings, self.input_character)
            char_embed = tf.reshape(char_embed, [-1, 6, embedding_size])
            # seq_len = tf.reshape(self.seq_len, [-1])
            # char_outputs, _ = self.bi_gru_encode(char_embed, seq_len)
            char_outputs = char_embed
            with tf.variable_scope('attention'):
                char_attn_outputs = self.attention(char_outputs, embedding_size)
                # char_attn_outputs = tf.reshape(char_attn_outputs, [self.batch_size, self.document_size, -1])

        # combine word embedding and character embedding
        with tf.variable_scope('combine'):
            embed = tf.add(word_embed, char_attn_outputs)*0.5
            embed = tf.reshape(embed, [-1, self.window_size, embedding_size])
            context_embed = tf.reduce_mean(embed, axis=1)
            self.save_embed = context_embed

        with tf.variable_scope('neg_loss'):
            nce_weights = tf.Variable(tf.random_uniform([word_size, embedding_size], -init, init))
            nce_biases = tf.Variable(tf.zeros([word_size]))
            # self.loss = tf.nn.nce_loss(weights=nce_weights,
            #                            biases=nce_biases,
            #                            labels=self.input_y,
            #                            inputs=context_embed,
            #                            num_sampled=num_sampled,
            #                            num_classes=word_size)

            sampled_values = candidate_sampling_ops.fixed_unigram_candidate_sampler(
                true_classes=self.input_y,
                num_true=1,
                num_sampled=num_sampled,
                unique=True,
                unigrams=dictionary_pro,
                range_max=word_size)
    
            self.loss = tf.nn.nce_loss(weights=nce_weights,
                                       biases=nce_biases,
                                       labels=self.input_y,
                                       inputs=context_embed,
                                       num_sampled=num_sampled,
                                       sampled_values=sampled_values,
                                       num_classes=word_size)

        self.train = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)

        valid_embeddings = tf.nn.embedding_lookup(word_embeddings, valid_dataset)
        self.similarity = tf.matmul(valid_embeddings, word_embeddings, transpose_b=True)

    def bi_gru_encode(self, inputs, sentence_size, scope=None):
        with tf.variable_scope(scope or 'bi_gru_encode') as scope:
            fw_cell = tf.nn.rnn_cell.GRUCell(self.embedding_size)
            bw_cell = tf.nn.rnn_cell.GRUCell(self.embedding_size)

            enc_out, (enc_state_fw, enc_state_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                                                    cell_bw=bw_cell,
                                                                                    inputs=inputs,
                                                                                    sequence_length=sentence_size,
                                                                                    dtype=tf.float32)

            enc_state = tf.concat(values=[enc_state_fw, enc_state_bw], concat_dim=1)
            enc_outputs = tf.concat(values=enc_out, concat_dim=2)
        return enc_outputs, enc_state

    def attention(self, inputs, size, scope=None):
        with tf.variable_scope(scope or 'attention') as scope:
            attention_context_vector = tf.get_variable(name='attention_context_vector',
                                                       shape=[size],
                                                       regularizer=layers.l2_regularizer(scale=L2_REG),
                                                       dtype=tf.float32)

            # input_projection = layers.fully_connected(inputs, size,
            #                                           activation_fn=tf.tanh,
            #                                           weights_regularizer=layers.l2_regularizer(scale=L2_REG))

            input_projection = inputs
            vector_attn = tf.reduce_sum(tf.multiply(input_projection, attention_context_vector), axis=2, keep_dims=True)
            attention_weights = tf.nn.softmax(vector_attn, dim=1)
            weighted_projection = tf.multiply(inputs, attention_weights)
            outputs = tf.reduce_sum(weighted_projection, axis=1)
            # outputs = layers.fully_connected(outputs, size,
            #                                  activation_fn=tf.tanh,
            #                                  weights_regularizer=layers.l2_regularizer(scale=L2_REG))
        return outputs

if __name__ == '__main__':
    model = WordCharModel(word_size=100, character_size=50, embedding_size=10, num_sampled=5)