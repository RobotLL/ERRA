#!/usr/bin/env bash

device=4
lm=(google/t5-base-lm-adapt)  #(google/t5-xl-lm-adapt  facebook/bart-large gpt2-xl google/t5-v1_1-xl)
declare -a versions=(50)  #50 250 1000 inf
dataset=(few_shot_comet) #
prompt_type=(prefix_plus) #prefix soft  prefix_plus
beam_num=(1)
declare -a num_prefixs=(5)
lr=(0.00005)
declare -a num_train_epochss=(3)
seed=(42)
declare -a tasks=(training)   #low1 low2 low3 low4 low4 low5 low6 low7 low8 low9 low10 low11 high1234 high2345 high678 high91011)  # cut drawer clean
declare -a levels=(500)   #training test training-RN_pre  training-RN_no_pre clip_saycan training-clip_saycan
#method=(ft) #prompt, ft  high low multi
for num_train_epochs in "${num_train_epochss[@]}"
do
    for num_prefix in "${num_prefixs[@]}"
    do
        for task in "${tasks[@]}"
        do
            for level in "${levels[@]}"
            do
#             #     if [[ $lm == *t5*lm* ]]; then   #[[ $lm == *t5* ]]
#             #         model="t5-lm"
#                 if [[ $version==50 ]]; then
#                     check_step=200
#                 elif [[ $version==250 ]]; then
#                     check_step=1000
#                 elif [[ $version==1000 ]]; then
#                     check_step=2500
#                 elif [[ $version==1000 ]]; then
#                     check_step=2500                    
#                 fi
                
                check_step=20000

                if [[ $lm == *t5* ]]; then   
                    model="t5"
                    max_length=15
                elif [[ $lm == *bart* ]]; then
                    model="bart"
                    max_length=15
                elif [[ $lm == *gpt2* ]]; then
                    model="gpt2"
                    max_length=30
                fi

#              Find available device
                while [ $device -gt 3 ]
                do
                    for ((i=0;i<=3;i++));
                    do
                        info=`nvidia-smi -i ${i}`
                        if [[ $info == *"No running processes found"* ]]; then
                            device=$i
                            echo "Using device ${device}"
                            break
                        fi
                    done
                    if [[ $device -gt 3 ]]; then
                        sleep 30s
                    fi
                done

                curr_device=${device};
                device=4;
#                 curr_device=2

                PYTHONIOENCODING=utf-8 python3   lm.py \
                    --device ${curr_device} \
                    --data_dir ../data/ \
                    --output_dir output/no-tact-cpu5-${model}-${task}-${level}/ \
                    --model ${model} \
                    --task ${task} \
                    --level ${level} \
                    --prompt_type ${prompt_type} \
                    --model_name_or_path ${lm} \
                    --plm_eval_mode \
                    --lr ${lr} \
                    --num_train_epochs ${num_train_epochs} \
                    --check_step ${check_step} \
                    --max_length 20 \
                    --min_length 1 \
                    --seed ${seed} \
                    --num_prefix ${num_prefix} \
                    --beam_num ${beam_num} \
                    --batch_size 16 \
                    --do_train \
                    --do_test \
                    --do_eval \
                    --using_decoder_past_key_values
                sleep 60s
            done
        done
    done
done
#output/${model}-${task}-high/ 
#                     --do_train \
#                     --do_eval \
#                     --test_init_path  ./output/t5-all/lr5e-05-ep3-sd42\