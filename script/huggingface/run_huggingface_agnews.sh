#!/bin/bash

DATASET=agnews
MODEL=ada
NSHOT=2
LOGDIR=experiment/${DATASET}

for SEED in 1 2 3 4 5;

do

  INPUT_FILE=`ls ${LOGDIR}/ckpt/generate_3gram_${DATASET}_${NSHOT}_shot_gpt2-xl_seed${SEED}_*`

  python huggingface_compeletion.py --ckpt ${INPUT_FILE} --model ${MODEL}

  HUGGINGFACE_RESP_FILE=`ls ${LOGDIR}/ckpt/generate_3gram_${DATASET}_${NSHOT}_shot_${MODEL}_seed${SEED}_*.pkl`

  python huggingface_augment.py ${HUGGINGFACE_RESP_FILE}

  TEST_FILE=`ls ${LOGDIR}/ckpt/augment_generate_3gram_${DATASET}_${NSHOT}_shot_${MODEL}_seed${SEED}_*.jsonl`

  python huggingface_prediction.py --config config/${DATASET}.yaml --nshot ${NSHOT} --model ${MODEL} --output ${LOGDIR}  --seed ${SEED} --test_data_path ${TEST_FILE}

  cd ${LOGDIR}

  FAKE_CKPT=`ls ${DATASET}_${NSHOT}_shot_${MODEL}_seed${SEED}_*.pkl`

  mv ${FAKE_CKPT} fake_${FAKE_CKPT}

  cd -

  python huggingface_prediction.py --config config/${DATASET}.yaml --nshot ${NSHOT} --model ${MODEL} --output ${LOGDIR}  --seed ${SEED}

  cd ${LOGDIR}

  TRUE_CKPT=`ls ${DATASET}_${NSHOT}_shot_${MODEL}_seed${SEED}_*.pkl`
  mv ${TRUE_CKPT} true_${TRUE_CKPT}

  cd -

  python entropy.py --true ${LOGDIR}/true_${TRUE_CKPT} --fake ${LOGDIR}/fake_${FAKE_CKPT} --topk 4 --save ${LOGDIR}/result_${DATASET}_${NSHOT}_shot_${MODEL}_seed${SEED}.json

done

