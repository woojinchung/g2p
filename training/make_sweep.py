# Create a script to run a random hyperparameter search.

import random
import numpy as np
import gflags
import sys
from datetime import datetime

SINGLE_DAY = False
now = datetime.now()
NAME = "{0[0]:02d}{0[1]:02d}{0[2]:02d}{0[3]:02d}{0[4]:02d}".format([now.month, now.day, now.hour, now.minute, now.second])
SWEEP_RUNS = 20

LIN = "LIN"
EXP = "EXP"
SS_BASE = "SS_BASE"
BOOL = "BOOL"
CHOICE = "CHOICE"
MUL = "MUL" # multiple of 100
QUAD = "QUAD"

FLAGS = gflags.FLAGS


gflags.DEFINE_string("data_dir", "cmu/no_stress/", "")
# gflags.DEFINE_string("vocab_path", "/scratch/asw462/data/bnc-30/vocab_20000.txt", "")
# gflags.DEFINE_string("embedding_path", "/scratch/asw462/data/bnc-30/embeddings_20000.txt", "")
gflags.DEFINE_string("log_path", "/logs/", "")

FLAGS(sys.argv)

# Instructions: Configure the variables in this block, then run
# the following on a machine with qsub access:
# python make_sweep.py > my_sweep.sh
# bash my_sweep.sh

# - #

# Non-tunable flags that must be passed in.

FIXED_PARAMETERS = {
    #paths
    # "data_dir": "/scratch/asw462/data/bnc_lm/",
    # "vocab_path": "/scratch/asw462/data/bnc-30/vocab_20000.txt",
    # "embedding_path": "/scratch/asw462/data/bnc-30/embeddings_20000.txt",
    "log_path": ".",
    # "data_type":     "discriminator",
    "model_type":      "ATTN",
    "ckpt_path":  "checkpoints/",
    # "gpu": "",

    #sizes
    # "embedding_size":   "300",
    "crop_pad_length": "20",


    #chunks
    "stages_per_epoch": "100",
    "prints_per_stage": "5",
    "convergence_threshold": "20",
    "max_epochs": "100",
    "batch_size": "64",


}

# Tunable parameters.
SWEEP_PARAMETERS = {

    "embedding_size": ("embedding", QUAD, 50, 300),
    "hidden_size": ("h_size", QUAD, 225, 1600),
    "num_layers": ("num_layers", LIN, 1, 4),
    "learning_rate": ("lr", EXP, 0.00005, 0.05),
    "data_dir": ("data", CHOICE, ["cmu/stress/", "cmu/no_stress/"], None)

}


sweep_name = "sweep_" + NAME + "_" + FIXED_PARAMETERS["model_type"]

# - #
print "# NAME: " + sweep_name
print "# NUM RUNS: " + str(SWEEP_RUNS)
print "# SWEEP PARAMETERS: " + str(SWEEP_PARAMETERS)
print "# FIXED_PARAMETERS: " + str(FIXED_PARAMETERS)
print
print "SWEEP=\"%s\" " % NAME
print "mkdir ~/logs/$SWEEP"
print

for run_id in range(SWEEP_RUNS):
    params = {}
    name = sweep_name + "_" + str(run_id)

    params.update(FIXED_PARAMETERS)
    # Any param appearing in both sets will be overwritten by the sweep value.

    for param in SWEEP_PARAMETERS:
        config = SWEEP_PARAMETERS[param]
        t = config[1]
        mn = config[2]
        mx = config[3]

        r = random.uniform(0, 1)
        if t == EXP:
            lmn = np.log(mn)
            lmx = np.log(mx)
            sample = np.exp(lmn + (lmx - lmn) * r)
        elif t == QUAD:
            sqmn = np.sqrt(mn)
            sqmx = np.sqrt(mx)
            sample = np.power(sqmn + (sqmx - sqmn) * r, 2)
        elif t == BOOL:
            sample = r > 0.5
        elif t==SS_BASE:
            lmn = np.log(mn)
            lmx = np.log(mx)
            sample = 1 - np.exp(lmn + (lmx - lmn) * r)
        elif t==CHOICE:
            sample = random.choice(mn)
        elif t==MUL:
            x = mn + (mx - mn) * r
            sample = x + 100 - x % 100
        else:
            sample = mn + (mx - mn) * r

        if isinstance(mn, int):
            sample = int(round(sample, 0))
            val_disp = str(sample)
            params[param] = sample
        elif isinstance(mn, float):
            val_disp = "%.2g" % sample
            params[param] = sample
        elif t==BOOL:
            val_disp = str(int(sample))
            if not sample:
                params['no' + param] = ''
            else:
                params[param] = ''
        else:
            val_disp = sample
            params[param] = sample
        if "/" in val_disp:
            val_disp = filter(lambda x: x != "", val_disp.split("/"))[-1]
        name += "-" + config[0] + val_disp

    flags = ""
    for param in params:
        value = params[param]
        flags += " --" + param + " " + str(value)
        if param == "es_num_roots":
            root = value
        elif param == "es_num_episodes":
            eps = value

    flags += " --experiment_name " + name

    if SINGLE_DAY:
        print "cd ~/g2p; python -m training.run_training" + flags
    else:
        print "export FLAGS=\"" + flags + "\""
        print "cd ~/g2p; sbatch my_sweep.sbatch"
    print
