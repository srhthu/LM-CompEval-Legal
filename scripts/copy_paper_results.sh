for model in gpt4 chatgpt bloomz_7B chatglm_6B vicuna_13B;
do
if [ ! -d "runs/paper_version/${model}" ]; then
  mkdir "runs/paper_version/${model}"
fi
for ttype in free multi
do
    for shot in {0..4}
    do
        [ $shot -eq 0 ] && set_suffix="" || set_suffix="_sim"
        cur_dir="runs/paper_version/${model}/${ttype}-${shot}shot"
        [ ! -d $cur_dir ] && mkdir $cur_dir
        echo $cur_dir
        cp "/storage_fast/rhshui/workspace/LJP/lm/runs/benchmark/${ttype}_${shot}shot${set_suffix}/${model}/raw_test_output.txt" "${cur_dir}/"
    done
    
done
done