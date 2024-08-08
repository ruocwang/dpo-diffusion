#!/bin/bash
prompt_path=${prompt_path:-"none"}
model=${model:-"gpt-4"}
name=${name:-"antonym"}
mode=${mode:-"default"}

while [ $# -gt 0 ]; do
    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
    fi
    shift
done

cd ../
python src/build_search_space.py \
    --prompt_path $prompt_path \
    --model $model \
    --space_name $name \
    --mode $mode
