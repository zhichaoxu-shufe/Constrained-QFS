
# dbpedia
CUDA_VISIBLE_DEVICES=3 python finetune/finetune_dbpedia.py \
--input_dir /home/sci/zhichao.xu/cg_dataset/dbpedia_processed \
--experiment_root ./ \
--model_name_or_path facebook/bart-large \
--tokenizer facebook/bart-large \
--input_format bart-concat \
--iterations 10 \
--save_ckpt

# pubmedqa
CUDA_VISIBLE_DEVICES=3 python finetune/finetune_pubmedqa.py \
--input_dir /home/sci/zhichao.xu/cg_dataset/pubmedqa_processed \
--experiment_root ./ \
--tokenizer facebook/bart-large \
--model_name_or_path facebook/bart-large \
--input_format bart-concat \
--iterations 10 \
--fp16 \
--save_ckpt \
--do_train \
--do_eval