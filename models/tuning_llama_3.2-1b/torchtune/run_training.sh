#!/bin/sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=FALSE


tune run lora_finetune_single_device --config ./COROLA_INSTRUCT_1B_qlora_single_device.yaml
