import gflags

def get_flags():
    # Debug settings.
    gflags.DEFINE_string("data_dir",
                         "cmu",
                         "dir containing train.txt, test.txt, valid.txt")
    gflags.DEFINE_string("log_path", "logs", "")
    gflags.DEFINE_string("data_type", "discriminator", "figure out how to use this")
    gflags.DEFINE_enum("model_type", "LSTM", ["LSTM", "BiLSTM", "DEEP", "ENC_DEC", "ATTN"], "options: LSTM, BiLSTM, ...")
    gflags.DEFINE_string("ckpt_path", "checkpoints", "")
    gflags.DEFINE_boolean("gpu", False, "set to false on local")
    gflags.DEFINE_string("experiment_name", "", "")

    #sizes
    gflags.DEFINE_integer("embedding_size", 44, "hardcoded for simplicity")
    gflags.DEFINE_integer("reduction_size", 70, "hardcoded for simplicity")
    gflags.DEFINE_integer("crop_pad_length", 30, "")

    #chunks
    gflags.DEFINE_integer("stages_per_epoch",
                          40,
                          "how many eval/stats steps per epoch?")
    gflags.DEFINE_integer("prints_per_stage",
                          1,
                          "how often to print stats to stdout during epoch")
    gflags.DEFINE_integer("convergence_threshold",
                          50,
                          "how many eval steps before early stop")
    gflags.DEFINE_integer("max_epochs",
                          100,
                          "number of epochs before stop, essentially unreachable")
    gflags.DEFINE_integer("batch_size", 64, "")

    #tunable parameters
    gflags.DEFINE_integer("hidden_size", 1024, "")
    gflags.DEFINE_integer("num_layers", 1, "")
    gflags.DEFINE_float("learning_rate", .002, "")
