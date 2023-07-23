
CUDA_VISIBLE_DEVICES=3 python decoding_seq2seq/decode.py \
--input_dir /home/sci/zhichao.xu/cg_dataset/dbpedia_processed \
--saliency_model_name_or_path /home/sci/zhichao.xu/saliency/saliency_model_cross_encoder/ \
--saliency_tokenizer distilbert-base-uncased \
--model_name_or_path /home/sci/zhichao.xu/saliency/experiment/bart_ckpt_dbpedia/bart-large_epoch_16/ \
--tokenizer facebook/bart-large \
--input_format bart-concat \
--topk_constraint 3 \
--sat_tolerance 3 \
--constraints_mode document-only \
--test_samples 200 