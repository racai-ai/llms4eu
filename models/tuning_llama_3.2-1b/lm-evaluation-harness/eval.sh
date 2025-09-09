#!/bin/sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

#lm_eval --model hf \
#    --model_args pretrained=../models/Meta-Llama-3.2-1B,parallelize=True,load_in_4bit=True,peft=../tuned/Meta-Llama-3.2-1B/lora_single_device/epoch_0 \
#    --tasks ro_sts \
#    --device cuda:0

#accelerate launch -m lm_eval --model hf \
#    --model_args pretrained=../models/Meta-Llama-3.2-1B-Instruct,parallelize=True,load_in_4bit=True,peft=../tuned/Meta-Llama-3.2-1B-Instruct/lora_single_device/epoch_3 \
#    --tasks ro_sts \
#    --device cuda:0 \
#    --output_path ./eval_Meta-Llama-3.2-1B-Instruct-ep3 \
#    --apply_chat_template --fewshot_as_multiturn --num_fewshot=1 \
#    --batch_size 32

    #--apply_chat_template --fewshot_as_multiturn --num_fewshot=1 \


#accelerate launch -m \
#    lm_eval --model hf \
#    --model_args pretrained=../models/Meta-Llama-3.2-1B-Instruct,parallelize=True,dtype=bfloat16,load_in_4bit=True,peft=../tuned/Meta-Llama-3.2-1B-Instruct/qlora_single_device/epoch_0 \
#    --tasks ro_sts \
#    --device cuda:0 \
#    --output_path ./eval_Meta-Llama-3.2-1B-Instruct-ep3 \
#    --apply_chat_template --fewshot_as_multiturn --num_fewshot=1 \
#    --batch_size 128


#    lm_eval --model hf \
#    --model_args pretrained=../tuned/COROLA-INSTRUCT-Meta-Llama-3.2-1B/merged/e1,parallelize=True,dtype=bfloat16,load_in_4bit=True \
#    --tasks ro_sts \
#    --device cuda:0 \
#    --output_path ./eval_COROLA-INSTRUCT-Meta-Llama-3.2-1B-e1 \
#    --apply_chat_template --fewshot_as_multiturn --num_fewshot=1 \
#    --batch_size 128

#    lm_eval --model hf \
#    --model_args pretrained=../tuned/COROLA-INSTRUCT-Meta-Llama-3.2-1B/merged/e1,parallelize=True,dtype=bfloat16,load_in_4bit=True \
#    --tasks ro_xquad \
#    --device cuda:0 \
#    --output_path ./eval_COROLA-INSTRUCT-Meta-Llama-3.2-1B-e1 \
#    --apply_chat_template \
#    --num_fewshot=0 \
#    --batch_size 4


#    lm_eval --model hf \
#    --model_args pretrained=../tuned/COROLA-Meta-Llama-3.2-1B/merged/e2,parallelize=True,dtype=bfloat16,load_in_4bit=True \
#    --tasks ro_arc_challenge \
#    --device cuda:0 \
#    --output_path ./eval_COROLA-Meta-Llama-3.2-1B-e2 \
#    --num_fewshot=0 \
#    --batch_size 4

#    --fewshot_as_multiturn \


#    lm_eval --model hf \
#    --model_args pretrained=../tuned/COROLA-Meta-Llama-3.2-1B/merged/e2,parallelize=True,dtype=bfloat16,load_in_4bit=True \
#    --tasks ro_xquad \
#    --device cuda:0 \
#    --output_path ./eval_COROLA-Meta-Llama-3.2-1B-e2 \
#    --num_fewshot=1 \
#    --batch_size 4


accelerate launch -m lm_eval --model hf \
    --model_args pretrained=../tuned/COROLA-Meta-Llama-3.2-1B/merged/e0,parallelize=True,load_in_4bit=True \
    --tasks ro_xquad \
    --device cuda:0 \
    --output_path ./eval_COROLA-Meta-Llama-3.2-1B-e0 \
    --num_fewshot=8 \
    --batch_size 4

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=../tuned/COROLA-Meta-Llama-3.2-1B/merged/e0,parallelize=True,load_in_4bit=True \
    --tasks ro_xquad \
    --device cuda:0 \
    --output_path ./eval_COROLA-Meta-Llama-3.2-1B-e0 \
    --num_fewshot=25 \
    --batch_size 4

#instruct e0
accelerate launch -m lm_eval --model hf \
    --model_args pretrained=../tuned/COROLA-INSTRUCT-Meta-Llama-3.2-1B/merged/e0,parallelize=True,load_in_4bit=True \
    --tasks ro_arc_challenge \
    --device cuda:0 \
    --output_path ./eval_COROLA-INSTRUCT-Meta-Llama-3.2-1B-e0 \
    --apply_chat_template \
    --num_fewshot=0 \
    --batch_size 4

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=../tuned/COROLA-INSTRUCT-Meta-Llama-3.2-1B/merged/e0,parallelize=True,load_in_4bit=True \
    --tasks ro_arc_challenge \
    --device cuda:0 \
    --output_path ./eval_COROLA-INSTRUCT-Meta-Llama-3.2-1B-e0 \
    --apply_chat_template \
    --num_fewshot=1 \
    --batch_size 4

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=../tuned/COROLA-INSTRUCT-Meta-Llama-3.2-1B/merged/e0,parallelize=True,load_in_4bit=True \
    --tasks ro_arc_challenge \
    --device cuda:0 \
    --output_path ./eval_COROLA-INSTRUCT-Meta-Llama-3.2-1B-e0 \
    --apply_chat_template \
    --num_fewshot=5 \
    --batch_size 4

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=../tuned/COROLA-INSTRUCT-Meta-Llama-3.2-1B/merged/e0,parallelize=True,load_in_4bit=True \
    --tasks ro_arc_challenge \
    --device cuda:0 \
    --output_path ./eval_COROLA-INSTRUCT-Meta-Llama-3.2-1B-e0 \
    --apply_chat_template \
    --num_fewshot=8 \
    --batch_size 4

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=../tuned/COROLA-INSTRUCT-Meta-Llama-3.2-1B/merged/e0,parallelize=True,load_in_4bit=True \
    --tasks ro_arc_challenge \
    --device cuda:0 \
    --output_path ./eval_COROLA-INSTRUCT-Meta-Llama-3.2-1B-e0 \
    --apply_chat_template \
    --num_fewshot=25 \
    --batch_size 4

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=../tuned/COROLA-INSTRUCT-Meta-Llama-3.2-1B/merged/e0,parallelize=True,load_in_4bit=True \
    --tasks ro_xquad \
    --device cuda:0 \
    --output_path ./eval_COROLA-INSTRUCT-Meta-Llama-3.2-1B-e0 \
    --apply_chat_template \
    --num_fewshot=0 \
    --batch_size 4

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=../tuned/COROLA-INSTRUCT-Meta-Llama-3.2-1B/merged/e0,parallelize=True,load_in_4bit=True \
    --tasks ro_xquad \
    --device cuda:0 \
    --output_path ./eval_COROLA-INSTRUCT-Meta-Llama-3.2-1B-e0 \
    --apply_chat_template \
    --num_fewshot=1 \
    --batch_size 4

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=../tuned/COROLA-INSTRUCT-Meta-Llama-3.2-1B/merged/e0,parallelize=True,load_in_4bit=True \
    --tasks ro_xquad \
    --device cuda:0 \
    --output_path ./eval_COROLA-INSTRUCT-Meta-Llama-3.2-1B-e0 \
    --apply_chat_template \
    --num_fewshot=5 \
    --batch_size 4

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=../tuned/COROLA-INSTRUCT-Meta-Llama-3.2-1B/merged/e0,parallelize=True,load_in_4bit=True \
    --tasks ro_xquad \
    --device cuda:0 \
    --output_path ./eval_COROLA-INSTRUCT-Meta-Llama-3.2-1B-e0 \
    --apply_chat_template \
    --num_fewshot=8 \
    --batch_size 4

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=../tuned/COROLA-INSTRUCT-Meta-Llama-3.2-1B/merged/e0,parallelize=True,load_in_4bit=True \
    --tasks ro_xquad \
    --device cuda:0 \
    --output_path ./eval_COROLA-INSTRUCT-Meta-Llama-3.2-1B-e0 \
    --apply_chat_template \
    --num_fewshot=25 \
    --batch_size 4



#    --tasks ro_sts --batch_size 32 \
