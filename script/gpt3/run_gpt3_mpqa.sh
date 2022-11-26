#!/bin/bash

DATASET=mpqa
MODEL=ada
NSHOT=4
LOGDIR=experiment/${DATASET}

for SEED in 1 2 3 4 5;

do

  INPUT_FILE=`ls ${LOGDIR}/ckpt/generate_3gram_${DATASET}_${NSHOT}_shot_gpt2-xl_seed${SEED}_*`

  python gpt3_compeletion.py --ckpt ${INPUT_FILE} --model ${MODEL}

  GPT3_RESP_FILE=`ls ${LOGDIR}/ckpt/generate_3gram_${DATASET}_${NSHOT}_shot_${MODEL}_seed${SEED}_*.pkl`

  python gpt3_augment.py ${GPT3_RESP_FILE}

  TEST_FILE=`ls ${LOGDIR}/ckpt/augment_generate_3gram_${DATASET}_${NSHOT}_shot_${MODEL}_seed${SEED}_*.jsonl`

  python gpt3_prediction.py --config config/${DATASET}.yaml --nshot ${NSHOT} --model ${MODEL} --output ${LOGDIR}  --seed ${SEED} --test_data_path ${TEST_FILE}

  cd ${LOGDIR}

  FAKE_CKPT=`ls ${DATASET}_${NSHOT}_shot_${MODEL}_seed${SEED}_*.pkl`

  mv ${FAKE_CKPT} fake_${FAKE_CKPT}

  cd -

  python gpt3_prediction.py --config config/${DATASET}.yaml --nshot ${NSHOT} --model ${MODEL} --output ${LOGDIR}  --seed ${SEED}

  cd ${LOGDIR}

  TRUE_CKPT=`ls ${DATASET}_${NSHOT}_shot_${MODEL}_seed${SEED}_*.pkl`
  mv ${TRUE_CKPT} true_${TRUE_CKPT}

  cd -

  python entropy.py --true ${LOGDIR}/true_${TRUE_CKPT} --fake ${LOGDIR}/fake_${FAKE_CKPT} --topk 4 --save ${LOGDIR}/result_${DATASET}_${NSHOT}_shot_${MODEL}_seed${SEED}.json

done

