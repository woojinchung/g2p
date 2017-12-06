import gflags
import models.rnn_classifier
import models.attention
import models.encoder_decoder
import my_flags
import sys
import torch

FLAGS = gflags.FLAGS

if __name__ == '__main__':
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

        if FLAGS.gpu:
            cl = cl.cuda()

        optimizer = torch.optim.Adam(cl.parameters(), lr=FLAGS.learning_rate)

        epochs = 0
        stages = 0
        best_dev_error = 1.

        try:
            checkpoint = torch.load(FLAGS.ckpt_path + "/" + FLAGS.experiment_name + ".ckpt")
            
            cl.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epochs = checkpoint['epochs']
            stages = checkpoint['stages']
            best_dev_error = checkpoint['best_dev_error']

            print "Resuming at stage: {} with best dev accuracy: {}".format(stages, 1. - best_dev_error)
        except IOError:
            pass
        
        clt = models.rnn_classifier.RNNTrainer(
            FLAGS,
            model=cl,
            optimizer=optimizer)
        clt.run(epochs, stages, best_dev_error)

    elif FLAGS.model_type == "ENC_DEC":
        cl = models.encoder_decoder.EncoderDecoder(
            hidden_size=FLAGS.hidden_size,
            embedding_size=FLAGS.embedding_size,
            output_size=FLAGS.reduction_size,
            num_layers=FLAGS.num_layers)

        if FLAGS.gpu:
            cl = cl.cuda()

        optimizer = torch.optim.Adam(cl.parameters(), lr=FLAGS.learning_rate)

        epochs = 0
        stages = 0
        best_dev_error = 1.

        try:
            checkpoint = torch.load(FLAGS.ckpt_path + "/" + FLAGS.experiment_name + ".ckpt")

            cl.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epochs = checkpoint['epochs']
            stages = checkpoint['stages']
            best_dev_error = checkpoint['best_dev_error']

            print "Resuming at stage: {} with best dev accuracy: {}".format(stages, 1. - best_dev_error)
        except IOError:
            pass

        clt = models.encoder_decoder.EDTrainer(
            FLAGS,
            model=cl,
            optimizer=optimizer)
        clt.run(epochs, stages, best_dev_error)
