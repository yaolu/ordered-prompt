#!/bin/bash

MAIN_DIR=$(pwd)
DATASET=dbpedia
LOGDIR=experiment/$DATASET;

DEVICE=3

mkdir -p $LOGDIR

for NSHOT in 1;
do

for MODEL in gpt2 gpt2-medium gpt2-large gpt2-xl gpt-neo-125M gpt-neo-1.3B gpt-neo-2.7B;
do
  for SEED in 1 2 3 4 5;
do
  for N in 3 5;
  do
     python main.py --config config/$DATASET.yaml \
     --nshot $NSHOT --model $MODEL --output $LOGDIR --seed $SEED \
     --ngram $N --generate --temperature 2.0 --topk 20 --do_sample --device $DEVICE

     echo "python main.py --config config/$DATASET.yaml \
     --nshot $NSHOT --model $MODEL --output $LOGDIR --seed $SEED \
     --ngram $N --generate --temperature 2.0 --topk 20 --do_sample --train_sample_mode random --device $DEVICE"
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
     --nshot $NSHOT --model $MODEL --output $LOGDIR --seed $SEED --test_data_path $LOGDIR/$OUTPUT --device $DEVICE #--train_sample_mode random

  mv $LOGDIR/${DATASET}_${NSHOT}_shot_${MODEL}_seed${SEED}_*.pkl $LOGDIR/fake_${DATASET}_${NSHOT}_shot_${MODEL}_seed${SEED}.pkl
  python main.py --config config/$DATASET.yaml \
     --nshot $NSHOT --model $MODEL --output $LOGDIR --seed $SEED --device $DEVICE #--train_sample_mode random
  mv $LOGDIR/${DATASET}_${NSHOT}_shot_${MODEL}_seed${SEED}_*.pkl $LOGDIR/true_${DATASET}_${NSHOT}_shot_${MODEL}_seed${SEED}.pkl

  python entropy.py --true $LOGDIR/true_${DATASET}_${NSHOT}_shot_${MODEL}_seed${SEED}.pkl \
                    --fake $LOGDIR/fake_${DATASET}_${NSHOT}_shot_${MODEL}_seed${SEED}.pkl \
                    --topk 4 --save $LOGDIR/result_${DATASET}_${NSHOT}_shot_${MODEL}_seed${SEED}.json

done;

done;

done;
