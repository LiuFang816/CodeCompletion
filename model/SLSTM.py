# -*- coding: utf-8 -*-
import tensorflow as tf
class LSTMConfig(object):
    embedding_size=64
    vocab_size=5000
    num_layer=1
    num_steps=400
    hidden_size=64
    dropout_keep_prob=0.8
    learning_rate=0.5
    batch_size=1
    num_epochs=50
    print_per_batch=1
    save_per_batch=1
    max_grad_norm=5
    rnn='lstm'

class LSTM(object):
    def __init__(self,config):
        self.config=config
        self.input=tf.placeholder(tf.int64,[None,self.config.num_steps-1],name='input')
        self.label=tf.placeholder(tf.int64,[None,self.config.num_steps-1],name='label')
        self.keep_prob=tf.placeholder(tf.float32,name='keep_prob')
        self.rnn()
    def rnn(self):
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(self.config.hidden_size,state_is_tuple=True)
        def gru_cell():
            return tf.contrib.rnn.GRUCell(self.config.hidden_size)
        def drop_out():
            if self.config.rnn=='lstm':
                cell=lstm_cell()
            else:
                cell=gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob=self.config.dropout_keep_prob)
        with tf.device('/cpu:0'):
            embedding=tf.get_variable('embedding',[self.config.vocab_size,self.config.embedding_size])
            embedding_inputs=tf.nn.embedding_lookup(embedding,self.input)
        with tf.name_scope('rnn'):
            cells=[drop_out() for _ in range(self.config.num_layer)]
            rnn_cell=tf.contrib.rnn.MultiRNNCell(cells,state_is_tuple=True)
            outputs,_=tf.nn.dynamic_rnn(rnn_cell,embedding_inputs,dtype=tf.float32)
            output = tf.reshape(tf.concat(outputs, 1), [-1, self.config.hidden_size])
            softmax_w = tf.get_variable(
                "softmax_w", [self.config.hidden_size, self.config.vocab_size], tf.float32)
            softmax_b = tf.get_variable("softmax_b", [self.config.vocab_size], tf.float32)
            logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
            # Reshape logits to be a 3-D tensor for sequence loss
            logits = tf.reshape(logits, [self.config.batch_size, self.config.num_steps-1, self.config.vocab_size])
            self.predict_class=tf.argmax(logits,2)
            # Use the contrib sequence loss and average over the batches
            loss = tf.contrib.seq2seq.sequence_loss(
                logits,
                self.label,
                tf.ones([self.config.batch_size, self.config.num_steps-1], dtype=tf.float32),
                average_across_timesteps=False,
                average_across_batch=True)
            # Update the cost
            self.loss = tf.reduce_sum(loss)

        with tf.name_scope('optimize'):
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                                              self.config.max_grad_norm)
            optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            self._train_op = optimizer.apply_gradients(
                zip(grads, tvars),
                global_step=tf.train.get_or_create_global_step())

        with tf.name_scope('accuracy'):
            correct_pre=tf.equal(self.label,self.predict_class)
            self.acc=tf.reduce_mean(tf.cast(correct_pre,tf.float32))


    def _build_rnn_graph_lstm(self, inputs, config, is_training):
        """Build the inference graph using canonical LSTM cells."""
    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
        def make_cell():
            cell = tf.contrib.rnn.BasicLSTMCell(
                config.hidden_size, forget_bias=0.0, state_is_tuple=True,
                reuse=not is_training)
            if is_training and config.keep_prob < 1:
                cell = tf.contrib.rnn.DropoutWrapper(
                    cell, output_keep_prob=config.keep_prob)
            return cell

        cell = tf.contrib.rnn.MultiRNNCell(
            [make_cell() for _ in range(config.num_layers)], state_is_tuple=True)

        self._initial_state = cell.zero_state(config.batch_size, data_type())
        state = self._initial_state
        # Simplified version of tensorflow_models/tutorials/rnn/rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        #
        # inputs = tf.unstack(inputs, num=num_steps, axis=1)
        # outputs, state = tf.contrib.rnn.static_rnn(cell, inputs,
        #                            initial_state=self._initial_state)
        outputs = []
        with tf.variable_scope("RNN"):
            for time_step in range(self.config.num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
        output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
        return output, state