#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_si284
dev_set=test_dev93
eval_sets="test_eval92 "

asr_config=conf/asr_train_transformer.yaml
pretrain_config=conf/pretrain_transformer.yaml

./pretrain.sh \
    --stage 12 \
    --stop_stage 100 \
    --nbpe 5000 \
    --ngpu 1 \
    --token_type char \
    --feats_type fbank_pitch \
    --use_lm false \
    --train_set "${train_set}" \
    --dev_set "${dev_set}" \
    --eval_sets "${eval_sets}" \
    --asr_config ${asr_config} \
    --pretrain_config "${pretrain_config}" \
    --srctexts "data/train_si284/text data/local/other_text/text" "$@"
