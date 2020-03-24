#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_si284
dev_set=test_dev93
eval_sets="test_eval92 "

asr_config=conf/train_transformer.yaml

ngpu=0

./pretrain.sh \
    --stage 10 \
    --stop_stage 10 \
    --nbpe 5000 \
    --ngpu ${ngpu} \
    --token_type char \
    --feats_type fbank_pitch \
    --use_lm false \
    --train_set "${train_set}" \
    --dev_set "${dev_set}" \
    --eval_sets "${eval_sets}" \
    --asr_config $asr_config \
    --srctexts "data/train_si284/text data/local/other_text/text" "$@"
