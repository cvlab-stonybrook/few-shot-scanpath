DATASET_NAME='COCO_Search'
MODEL_NAME='baseline'


CUDA_VISIBLE_DEVICES=0 python src/train.py \
  --log_root src/assets/TP-ex-789 \
  --fix_dir src/data/TP_fixations.json \
  --max_length 10 \
  --seed 10 --epoch 40 --start_rl_epoch 20 \
  --subject_feature_dim 384 \
  --no_eval_epoch 20 \
  --ex_subject 7 8 9 \
  --subject_num 7 \
  --batch 4 \
  --user_emb_path ../../../SE-Net/assets/TP-ex-789/train_user_embedding.pt