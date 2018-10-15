import os
import tensorflow as tf
import sys
from tensorflow.python.training.saver import latest_checkpoint
from config import *
from objective import get_optimization_ops, define_objective
from model import *
import model_and_data_serialization
sys.path.append(os.getcwd())
import numpy as np
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS

def evaluate(seq_length, N, charmap):

    lines, _, _ = model_and_data_serialization.load_dataset(seq_length=seq_length, b_charmap=False, b_inv_charmap=False,
                                                            n_examples=FLAGS.MAX_N_EXAMPLES, dataset='heldout')

    real_inputs_discrete = tf.placeholder(tf.int32, shape=[BATCH_SIZE, seq_length])

    global_step = tf.Variable(0, trainable=False)
    disc_cost, gen_cost, train_pred, train_pred_for_eval, disc_fake, disc_real, disc_on_inference, inference_op = define_objective(charmap,
                                                                                                            real_inputs_discrete,
                                                                                                            seq_length)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    # load checkpoints
    internal_checkpoint_dir = model_and_data_serialization.get_internal_checkpoint_dir(0)
    model_and_data_serialization.optimistic_restore(sess,
                                                    latest_checkpoint(internal_checkpoint_dir, "checkpoint"))
    restore_config.set_restore_dir(
        load_from_curr_session=True)  # global param, always load from curr session after finishing the first seq

    cnt = 0
    Linf_norm_values = []
    N_graph_values = []

    print("N = %d\n" % (N))
    # train_pred -> run session
    train_pred_all = np.zeros([N, BATCH_SIZE, seq_length, train_pred.shape[2]],
                            dtype=np.float32)

    for start_line in range(1):
        _data = np.array([[charmap[c] for c in l] for l in lines[start_line:start_line + BATCH_SIZE]])

        # rand N noise vectors and for each one - calculate train_pred.
        for i in range(N):
            train_pred_i = sess.run(train_pred_for_eval, feed_dict={real_inputs_discrete: _data})
            train_pred_all[i,:,:,:] = train_pred_i

    N_run = range(10, N+10, 10)
    train_pred_average_prev = np.mean(train_pred_all[0:N_run[0]], axis=0)

    for N_curr in N_run[1:]:
        N_graph_values.append(N_curr)
        train_pred_average = np.mean(train_pred_all[0:N_curr], axis=0)
        train_pred_average_diff = train_pred_average - train_pred_average_prev
        Linf_norm_diff = np.linalg.norm(train_pred_average_diff, axis=2, ord=np.inf)
        Linf_norm_values.append(np.mean(Linf_norm_diff))
        cnt += 1
        train_pred_average_prev = train_pred_average

    plt.figure()
    plt.plot(N_graph_values, Linf_norm_values)
    plt.title('Approximated error between empiric and real mean (Linf norm)', size=30)
    plt.xlabel('Number of random noise vectors (N)', size=16)
    plt.ylabel('Error (L_inf norm)', size=16)

def get_internal_checkpoint_dir(seq_length):
    internal_checkpoint_dir = os.path.join(restore_config.get_restore_dir(), "seq-%d" % seq_length)
    if not os.path.isdir(internal_checkpoint_dir):
        os.makedirs(internal_checkpoint_dir)
    return internal_checkpoint_dir

_, charmap, _ = model_and_data_serialization.load_dataset(seq_length=32, b_lines=False)
eval_seq_length = 7
N_max = 3000
evaluate(eval_seq_length, N_max, charmap)