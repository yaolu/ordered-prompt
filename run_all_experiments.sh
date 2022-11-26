#!/bin/bash

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024,garbage_collection_threshold:0.01'

# Fist Batch of Experiments
script/run_agnews.sh
script/run_cb.sh
script/run_cr.sh
script/run_dbpedia.sh
script/run_mpqa.sh
script/run_mr.sh
script/run_rte.sh
script/run_sst2.sh
script/run_sst5.sh
script/run_subj.sh
script/run_trec.sh
