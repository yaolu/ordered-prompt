#!/bin/bash

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024,garbage_collection_threshold:0.01'

MAIN_DIR=$(pwd)
DATASET=rte
LOGDIR=experiment/$DATASET;
SAMPLE_MODE=balance

DEVICE=6

mkdir -p $LOGDIR

for NSHOT in 4;
do

for MODEL in gpt2 gpt2-medium gpt2-large gpt2-xl gpt-neo-125M gpt-neo-1.3B gpt-neo-2.7B sharded-gpt-j-6B;
do
for SEED in 1 2 4;
do

  for N in 3 5;
  do
     python main.py --config config/$DATASET.yaml \
     --nshot $NSHOT --model $MODEL --output $LOGDIR --seed $SEED \
     --ngram $N --generate --temperature 0.2 --topk 20 --do_sample --device $DEVICE

     echo "python main.py --config config/$DATASET.yaml \
     --nshot $NSHOT --model $MODEL --output $LOGDIR --seed $SEED \
     --ngram $N --generate --temperature 0.2 --topk 20 --do_sample --device $DEVICE"
  done;

  cd $LOGDIR;

  for f in generate*;
  do
      python "$MAIN_DIR"/augment.py $f
  done;

  mkdir ckpt;
  mv *.pkl ckpt;

  OUTPUT=dev_${DATASET}_${NSHOT}_shot_${MODEL}_seed${SEED}.jsonl
  cat augment_*.jsonl > $OUTPUT
  mv augment_*.jsonl ckpt

  cd "${MAIN_DIR}" || exit;

  echo "python main.py --config config/$DATASET.yaml \
     --nshot $NSHOT --model $MODEL --output $LOGDIR --seed $SEED --test_data_path $LOGDIR/$OUTPUT --device $DEVICE;"

  python main.py --config config/$DATASET.yaml \
     --nshot $NSHOT --model $MODEL --output $LOGDIR --seed $SEED --test_data_path $LOGDIR/$OUTPUT --device $DEVICE

  mv $LOGDIR/${DATASET}_${NSHOT}_shot_${MODEL}_seed${SEED}_*.pkl $LOGDIR/fake_${DATASET}_${NSHOT}_shot_${MODEL}_seed${SEED}.pkl
  python main.py --config config/$DATASET.yaml \
     --nshot $NSHOT --model $MODEL --output $LOGDIR --seed $SEED --device $DEVICE
  mv $LOGDIR/${DATASET}_${NSHOT}_shot_${MODEL}_seed${SEED}_*.pkl $LOGDIR/true_${DATASET}_${NSHOT}_shot_${MODEL}_seed${SEED}.pkl

  python entropy.py --true $LOGDIR/true_${DATASET}_${NSHOT}_shot_${MODEL}_seed${SEED}.pkl \
                    --fake $LOGDIR/fake_${DATASET}_${NSHOT}_shot_${MODEL}_seed${SEED}.pkl \
                    --topk 4 --save $LOGDIR/result_${DATASET}_${NSHOT}_shot_${MODEL}_seed${SEED}.json

done;

done;

done;
