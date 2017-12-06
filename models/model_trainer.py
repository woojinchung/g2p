import torch
import gflags
from datetime import datetime
from torch.autograd import Variable
from utils import classifier_data_utils as cdu
import torch.nn.functional as F
import math

class ModelTrainer(object):
    def __init__(self, FLAGS, model, optimizer):
        self.FLAGS = FLAGS
        self.model = model
        self.optimizer = optimizer
        self.dm = cdu.DataManager(self.FLAGS.data_dir, self.FLAGS.embedding_size)
        self.loss = torch.nn.CrossEntropyLoss()
        self.OUTPUT_PATH = FLAGS.ckpt_path + "/" + FLAGS.experiment_name + ".ckpt"
        self.LOGS_PATH = FLAGS.log_path + "LOGS-" + FLAGS.experiment_name
        self.OUT_LOGS_PATH = FLAGS.log_path + "OUTPUTS-" + FLAGS.experiment_name
        self.LOGS = open(self.LOGS_PATH, "a")
        self.OUT_LOGS = open(self.OUT_LOGS_PATH, "a")

    def to_string(self):
        return "data\t\t\t" + self.FLAGS.data_dir + "\n" + \
            "input size\t\t" + str(self.FLAGS.embedding_size) + "\n" + \
            "hidden size\t\t" + str(self.model.hidden_size) + "\n" + \
            "learning rate\t" + str(self.FLAGS.learning_rate) + "\n" + \
            "output\t\t\t" + str(self.OUTPUT_PATH)


    def backprop(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def print_min_and_max(self, outputs, batch):
        max_prob, max_i_sentence = torch.topk(outputs.data, 1, 0)
        min_prob, min_i_sentence = torch.topk(outputs.data * -1, 1, 0)
        max_sentence = batch.sentences_view[max_i_sentence[0][0]]
        min_sentence = batch.sentences_view[min_i_sentence[0][0]]
        print("max:", max_prob[0][0], max_sentence)
        print("min:", min_prob[0][0] * -1, min_sentence)

    def print_stats(self, loss):
        # print("avg loss\t" + self.my_round(loss))
        print("avg loss:\t" + str(loss.data.numpy()[0]))

    def logs(self, n_batches, train_avg_loss, valid_avg_loss, t_confusion, v_confusion, model_saved):
        self.LOGS.write("\t" + str(n_batches) + "\t")
        self.LOGS.write("\t" + self.my_round(train_avg_loss) + "\t")
        self.LOGS.write("\t" + self.my_round(valid_avg_loss) + "\t")
        self.LOGS.write("\t" + self.my_round(t_confusion.matthews()) + "\t")
        self.LOGS.write("\t" + self.my_round(v_confusion.matthews()) + "\t")
        self.LOGS.write("\t" + self.my_round(t_confusion.f1()) + "\t")
        self.LOGS.write("\t" + self.my_round(v_confusion.f1()) + "\t")
        self.LOGS.write("\t" + "tp={0[0]:.4g}, tn={0[1]:.4g}, fp={0[2]:.4g}, fn={0[3]:.4g}".format(v_confusion.percentages()) + "\t")
        self.LOGS.write("\t" + str(model_saved) + "\n")
        self.LOGS.flush()

    def cluster_logs(self, n_batches, train_avg_loss, valid_avg_loss, t_confusion, v_confusion, model_saved):
        self.LOGS.write("\t" + str(n_batches))
        self.LOGS.write("\t" + self.my_round(train_avg_loss) + "\t")
        self.LOGS.write("\t" + self.my_round(valid_avg_loss) + "\t")
        self.LOGS.write("\t" + self.my_round(t_confusion.matthews()) + "\t")
        self.LOGS.write("\t" + self.my_round(v_confusion.matthews()) + "\t")
        self.LOGS.write("\t" + self.my_round(t_confusion.f1()))
        self.LOGS.write("\t" + self.my_round(v_confusion.f1()) + "\t")
        self.LOGS.write("\t" + "tp={0[0]:.2g}, tn={0[1]:.2g}, fp={0[2]:.2g}, fn={0[3]:.2g}".format(v_confusion.percentages()) + "\t")
        self.LOGS.write("\t" + str(model_saved) + "\n")
        self.LOGS.flush()

    @staticmethod
    def my_round(n):
        return "{0:.4g}".format(n)

    def get_metrics(self, outputs_list, labels_list):
        if self.FLAGS.gpu:
            outputs_list = [output.cuda() for output in outputs_list]
            labels_list = [label.cuda() for label in labels_list]

        loss = 0
        correct = 0
        preds = 0

        for i in range(len(outputs_list)):
            # calculate loss
            logits = F.log_softmax(outputs_list[i])
            loss += self.loss(logits, labels_list[i])

            # calculate class accuracy
            pred = logits.data.max(1)[1].cpu() # get the index of the max log-probability
            correct += pred.eq(labels_list[i].cpu().data).sum()
            preds += labels_list[i].size(0)

        return loss, correct, preds

    def run_batch(self, source_batch, labels_list, backprop):
        if self.FLAGS.gpu:
            input = source_batch[0].cuda()
        else:
            input = source_batch[0].cpu()

        outputs_list = self.model.forward(input, source_batch[1])
        loss, correct, preds = self.get_metrics(outputs_list, labels_list)

        if backprop:
            self.backprop(loss)
        if self.FLAGS.gpu:
            loss = loss.cpu()

        return loss, correct, preds

    def run_stage(self, epoch, backprop, stages_per_epoch, prints_per_stage):
        has_next = True
        n_batches = 0
        stage_batches = math.ceil(epoch.n_batches/stages_per_epoch)
        print_batches = int(math.ceil(stage_batches/prints_per_stage))

        print_loss = 0
        stage_loss = 0

        total_correct = 0
        total_pred = 0

        while has_next and n_batches < stage_batches:
            n_batches += 1
            source_batch, target_batch, has_next = epoch.get_new_batch()
            loss, correct, preds = self.run_batch(source_batch, target_batch, backprop)
            
            print_loss += loss
            stage_loss += loss

            total_correct += correct
            total_pred += preds

            if n_batches % print_batches == 0:
                self.print_stats(print_loss/print_batches)
                print_loss = 0
        if prints_per_stage > 1:
            self.print_stats(stage_loss/n_batches)

        class_acc = total_correct / float(total_pred)

        print "class_acc:\t" + str(class_acc)

        return stage_loss/n_batches, class_acc


    def run_epoch(self, n_stages_not_converging, epochs, n_stages, best_dev_err):
        train = cdu.CorpusEpoch(self.dm.training_pairs, self.dm, self.FLAGS.batch_size)
        valid = cdu.CorpusEpoch(self.dm.valid_pairs, self.dm, self.FLAGS.batch_size)
        for _ in range(self.FLAGS.stages_per_epoch):
            # if n_stages_not_converging > self.FLAGS.convergence_threshold:
            #     raise NotConvergingError
            n_stages += 1
            print("-------------training-------------")
            train_loss, train_acc = self.run_stage(train, True, self.FLAGS.stages_per_epoch, self.FLAGS.prints_per_stage)
            print("-------------validation-------------")
            valid_loss, valid_acc = self.run_stage(valid, False, self.FLAGS.stages_per_epoch, 1)

            if (1 - valid_acc) < 0.99 * best_dev_err and n_stages > 10:
                best_dev_err = 1 - valid_acc
                if self.FLAGS.gpu:
                    recursively_set_device(self.model.state_dict(), gpu=-1)
                    recursively_set_device(self.optimizer.state_dict(), gpu=-1)

                n_stages_not_converging = 0
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epochs': epochs,
                    'stages': n_stages,
                    'best_dev_error': best_dev_err,}, self.OUTPUT_PATH)

                if self.FLAGS.gpu:
                    recursively_set_device(self.model.state_dict(), gpu=0)
                    recursively_set_device(self.optimizer.state_dict(), gpu=0)

                print "Checkpointing with new best dev accuracy of " + str(valid_acc)
            else:
                n_stages_not_converging += 1

        return n_stages_not_converging, n_stages, best_dev_err

    def start_up_print_and_logs(self):
        print("======================================================================")
        print("                              TRAINING")
        print("======================================================================")
        print(self.to_string())
        self.LOGS.write("\n\n" + self.to_string() + "\n")
        self.LOGS.write(
            "# batches | train avg loss | valid avg loss | t matthews | v matthews | t f1 | v f1 |      confusion      |model saved\n" +
            "----------|----------------|----------------|------------|------------|------|------|---------------------|-----------\n")
        self.LOGS.flush()


    def run(self, epochs, stages, best_dev_err):
        """The outer loop of the model trainer"""
        self.start_up_print_and_logs()
        n_epoch = epochs
        n_stages = stages
        n_stages_not_converging = 0
        try:
            while n_epoch < self.FLAGS.max_epochs:
                print("===========================EPOCH %d=============================" % n_epoch)
                n_stages_not_converging, n_stages, best_dev_err = self.run_epoch(n_stages_not_converging, n_epoch, n_stages, best_dev_err)
                n_epoch += 1
        except NotConvergingError:
            # TODO: update
            self.model.load_state_dict(torch.load(self.OUTPUT_PATH))
            print("=====================TEST==================")
            test_loss = self.run_stage(cdu.CorpusEpoch(self.dm.test_pairs, self.dm), False, 1, 1)
#            self.LOGS.write("accuracy\t" + self.my_round(test_confusion.accuracy()) + "\n")
        except KeyboardInterrupt:
            print "\nSO LONG, AND THANKS FOR ALL THE FISH"
        finally:
            self.LOGS.close()


class NotConvergingError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

def recursively_set_device(inp, gpu=-1):
    if hasattr(inp, 'keys'):
        for k in inp.keys():
            inp[k] = recursively_set_device(inp[k], gpu)
    elif hasattr(inp, 'cpu'):
        if gpu >= 0:
            inp = inp.cuda()
        else:
            inp = inp.cpu()
    return inp
