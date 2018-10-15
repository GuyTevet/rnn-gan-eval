import os
import tensorflow as tf
import language_helpers
import sys
from tensorflow.python.training.saver import latest_checkpoint
from config import *
from language_helpers import generate_argmax_samples_and_gt_samples, inf_train_gen, decode_indices_to_string
from objective import get_optimization_ops, define_objective
from summaries import define_summaries, \
    log_samples
from model import *
import model_and_data_serialization
sys.path.append(os.getcwd())
import numpy as np
import time

FLAGS = tf.app.flags.FLAGS

def evaluate(seq_length, N, charmap, inv_charmap):

    lines, _, _ = model_and_data_serialization.load_dataset(seq_length=seq_length, b_charmap=False, b_inv_charmap=False,
                                                            n_examples=FLAGS.MAX_N_EXAMPLES, dataset='heldout')

    real_inputs_discrete = tf.placeholder(tf.int32, shape=[BATCH_SIZE, seq_length])

    global_step = tf.Variable(0, trainable=False)
    disc_cost, gen_cost, train_pred, train_pred_for_eval, disc_fake, disc_real, disc_on_inference, inference_op = define_objective(charmap,
                                                                                                            real_inputs_discrete,
                                                                                                            seq_length)

    # train_pred -> run session
    train_pred_all = np.zeros([N, BATCH_SIZE, seq_length, train_pred.shape[2]],
                            dtype=np.float32)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    # load checkpoints
    internal_checkpoint_dir = model_and_data_serialization.get_internal_checkpoint_dir(0)
    model_and_data_serialization.optimistic_restore(sess,
                                                    latest_checkpoint(internal_checkpoint_dir, "checkpoint"))
    restore_config.set_restore_dir(
        load_from_curr_session=True)  # global param, always load from curr session after finishing the first seq

    BPC_list = []

    for start_line in range(0, len(lines) - BATCH_SIZE + 1, BATCH_SIZE):
        t0 = time.time()
        _data = np.array([[charmap[c] for c in l] for l in lines[start_line:start_line + BATCH_SIZE]])

        # rand N noise vectors and for each one - calculate train_pred.
        for i in range(N):
            train_pred_i = sess.run(train_pred_for_eval, feed_dict={real_inputs_discrete: _data})
            train_pred_all[i,:,:,:] = train_pred_i

        # take average on each time step (first dimension)
        train_pred_average = np.mean(train_pred_all, axis=0)

        # compute BPC (char-based perplexity)
        train_pred_average_2d = train_pred_average.reshape([train_pred_average.shape[0]*train_pred_average.shape[1],
                                                            train_pred_average.shape[2]])
        real_data = _data.reshape([_data.shape[0]*_data.shape[1]])

        BPC = 0

        epsilon = 1e-20
        for i in range(real_data.shape[0]):
            BPC -= np.log2(train_pred_average_2d[i,real_data[i]]+epsilon)

        BPC /= real_data.shape[0]
        print("BPC of start_line %d/%d = %.2f" % (start_line, len(lines), BPC))
        print("t_iter = %.2f" % (time.time()-t0))
        BPC_list.append(BPC)
        np.save('BPC_list_temp.npy', BPC_list)

    BPC_final = np.mean(BPC_list)
    print("BPC_final = %.2f\n" % (BPC_final))
    np.save('BPC_list.npy', BPC_list)
    np.save('BPC_final.npy', BPC_final)

def get_internal_checkpoint_dir(seq_length):
    internal_checkpoint_dir = os.path.join(restore_config.get_restore_dir(), "seq-%d" % seq_length)
    if not os.path.isdir(internal_checkpoint_dir):
        os.makedirs(internal_checkpoint_dir)
    return internal_checkpoint_dir

_, charmap, inv_charmap = model_and_data_serialization.load_dataset(seq_length=32, b_lines=False)
eval_seq_length = 7
N = 2000
evaluate(eval_seq_length, N, charmap, inv_charmap)