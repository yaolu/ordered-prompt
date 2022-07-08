#!/bin/bash

MAIN_DIR=$(pwd)
DATASET=mpqa
LOGDIR=experiment/$DATASET;
#NSHOT=2
SAMPLE_MODE=balance

mkdir -p $LOGDIR

for NSHOT in 4;
do

for MODEL in gpt2 gpt2-medium gpt2-large gpt2-xl;
do
for SEED in 1 2 3 4 5;
do

  for N in 3 5;
  do
     python main.py --config config/$DATASET.yaml \
     --nshot $NSHOT --model $MODEL --output $LOGDIR --seed $SEED \
     --ngram $N --generate --temperature 2.0 --topk 20 --do_sample

     echo "python main.py --config config/$DATASET.yaml \
     --nshot $NSHOT --model $MODEL --output $LOGDIR --seed $SEED \
     --ngram $N --generate --temperature 2.0 --topk 20 --do_sample"
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
     --nshot $NSHOT --model $MODEL --output $LOGDIR --seed $SEED --test_data_path $LOGDIR/$OUTPUT;"

  python main.py --config config/$DATASET.yaml \
     --nshot $NSHOT --model $MODEL --output $LOGDIR --seed $SEED --test_data_path $LOGDIR/$OUTPUT

  mv $LOGDIR/${DATASET}_${NSHOT}_shot_${MODEL}_seed${SEED}_*.pkl $LOGDIR/fake_${DATASET}_${NSHOT}_shot_${MODEL}_seed${SEED}.pkl
  python main.py --config config/$DATASET.yaml \
     --nshot $NSHOT --model $MODEL --output $LOGDIR --seed $SEED
  mv $LOGDIR/${DATASET}_${NSHOT}_shot_${MODEL}_seed${SEED}_*.pkl $LOGDIR/true_${DATASET}_${NSHOT}_shot_${MODEL}_seed${SEED}.pkl

  python entropy.py --true $LOGDIR/true_${DATASET}_${NSHOT}_shot_${MODEL}_seed${SEED}.pkl \
                    --fake $LOGDIR/fake_${DATASET}_${NSHOT}_shot_${MODEL}_seed${SEED}.pkl \
                    --topk 4 --save $LOGDIR/result_${DATASET}_${NSHOT}_shot_${MODEL}_seed${SEED}.json

done;

done;

done;