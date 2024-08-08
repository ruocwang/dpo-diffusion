#!/bin/bash
#### exp management
script_name=`basename "$0"`
id=${script_name%.*}
gpu=${gpu:-"auto"}
override=${override:-""}

#### model
version=${version:-"v1-4"}

#### data
path=${path:-"data_submit/coco-v1-2s/usps-loss-above=0.85-100-improve-gpt-4.json"}
prompt_id=${prompt_id:-0}
num_seeds=${num_seeds:-1}

## config
algo_config=${algo_config:-"hybrid.yaml"}
task_config=${task_config:-"improve-antonym.yaml"}


while [ $# -gt 0 ]; do
    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
    fi
    shift
done
prompt_id_list=$(echo $prompt_id | tr ',' ' ')


cd ../
run_exp() {
    python src/run_dpo_diff.py \
        --save $id --gpu $gpu \
        --path $path --prompt_id $1 --version $version --num_seeds=$num_seeds \
        --algo_config=$algo_config --task_config=$task_config
}

for pid in $prompt_id_list
do
    if [ "$override" != "" ]
    then
        echo "$override" | run_exp $pid
    else
        run_exp $pid
    fi
done
