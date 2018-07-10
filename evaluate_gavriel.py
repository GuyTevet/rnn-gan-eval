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

FLAGS = tf.app.flags.FLAGS

def evaluate(seq_length, charmap, inv_charmap):

    lines, _, _ = model_and_data_serialization.load_dataset(seq_length=seq_length, b_charmap=False, b_inv_charmap=False,
                                                            n_examples=FLAGS.MAX_N_EXAMPLES)

    real_inputs_discrete = tf.placeholder(tf.int32, shape=[BATCH_SIZE, seq_length])

    global_step = tf.Variable(0, trainable=False)
    disc_cost, gen_cost, train_pred, disc_fake, disc_real, disc_on_inference, inference_op = define_objective(charmap,
                                                                                                            real_inputs_discrete,
                                                                                                            seq_length)
    # train_pred -> run session



_, charmap, inv_charmap = model_and_data_serialization.load_dataset(seq_length=32, b_lines=False)
eval_seq_length = 32  # fixme - change to a flag
evaluate(eval_seq_length, charmap, inv_charmap)

# if FLAGS.SCHEDULE_SPEC == 'all' :
#     stages = list(range(FLAGS.START_SEQ, FLAGS.END_SEQ))
# else:
#     split = FLAGS.SCHEDULE_SPEC.split(',')
#     stages = list(map(int, split))
#
# for i in range(len(stages)):
#     prev_seq_length = stages[i-1] if i>0 else 0
#     seq_length = stages[i]
#     tf.reset_default_graph()
#     if FLAGS.SCHEDULE_ITERATIONS:
#         iterations = min((seq_length + 1) * FLAGS.SCHEDULE_MULT, FLAGS.ITERATIONS_PER_SEQ_LENGTH)
#     else:
#         iterations = FLAGS.ITERATIONS_PER_SEQ_LENGTH
#
