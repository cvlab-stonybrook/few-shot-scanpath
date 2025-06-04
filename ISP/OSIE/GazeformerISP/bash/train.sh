# The name of this experiment.
DATASET_NAME='OSIE'
MODEL_NAME='baseline'

CUDA_VISIBLE_DEVICES=5 python src/train.py \
  --log_root /data/add_disk4/ruoyu/project/ISP/assets/git-osie-useremb-ex-10to15 \
  --subject_feature_dim 384 \
  --batch 4 \
  --epoch 40 \
  --no_eval_epoch 30 \
  --ex_subject 10 11 12 13 14 \
  --subject_num 10 \
  --user_emb_path /data/add_disk4/ruoyu/project/phat/git-osie-useremb-ex-10to15/train_user_embedding.pt


