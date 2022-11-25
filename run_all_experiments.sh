#!/bin/bash

# Fist Batch of Experiments
# GPUs 0-7
script/run_agnews.sh & 
script/run_cb.sh &  
script/run_cr.sh & 
script/run_dbpedia.sh &
script/run_mpqa.sh &
script/run_mr.sh &
script/run_rte.sh &
script/run_sst2.sh &

# # Second batch of Experiments
# # GPUs 0-2
# script/run_sst5.sh &
# script/run_subj.sh &
# script/run_trec.sh &
