import sys
import gflags
import my_flags
import models.rnn_classifier
import torch

FLAGS = gflags.FLAGS

if __name__ == '__main__':
    print("HELLO WORLD")
    my_flags.get_flags()

    # Parse command line flags.
    FLAGS(sys.argv)

    # flag_defaults(FLAGS)

    cl = None

    if FLAGS.model_type == "LSTM":
        cl = models.rnn_classifier.Classifier(
            hidden_size=FLAGS.hidden_size,
            embedding_size=FLAGS.embedding_size,
            reduction_size=FLAGS.reduction_size,
            num_layers=FLAGS.num_layers)
        try:
            cl.load_state_dict(torch.load(FLAGS.ckpt_path + FLAGS.experiment_name))
        except IOError:
            pass
        clt = models.rnn_classifier.RNNTrainer(
            FLAGS,
            model=cl)
        clt.run()
