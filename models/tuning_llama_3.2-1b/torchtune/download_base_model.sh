#!/bin/sh

HFTOKEN=$(head -n 1 ../hf_token.txt)

tune download meta-llama/Llama-3.2-1B \
    --output-dir ./Meta-Llama-3.2-1B \
    --ignore-patterns "original/consolidated.00.pth" \
    --hf-token $HFTOKEN


