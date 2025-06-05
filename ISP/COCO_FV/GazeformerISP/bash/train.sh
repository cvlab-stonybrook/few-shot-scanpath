# The name of this experiment.
DATASET_NAME='COCO_Search'
MODEL_NAME='baseline'


CUDA_VISIBLE_DEVICES=0 python src/train.py \
  --log_root src/assets/FV-ex-012 \
  --seed 10 --epoch 40 --start_rl_epoch 20 \
  --subject_feature_dim 384 \
  --no_eval_epoch 18 \
  --ex_subject 0 1 2 \
  --subject_num 7 \
  --batch 2 \
  --user_emb_path ../../../SE-Net/assets/FV-ex-012/train_user_embedding.pt