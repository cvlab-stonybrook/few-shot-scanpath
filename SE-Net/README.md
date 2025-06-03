# few-shot-scanpath

Subject Embedding Network

#### Installation
 - Use the installation in HAT[1] https://github.com/cvlab-stonybrook/HAT?tab=readme-ov-file:
 - 1) Install [Detectron2](https://github.com/facebookresearch/detectron2)
 - 2) Install MSDeformableAttn:
   ```
   cd ./hat/pixel_decoder/ops
   sh make.sh
   ```
#### Data
- Download images for COCO-Search18 and COCO-FreeView from https://drive.google.com/drive/folders/1im5SJKQ976MmB7JeRgtMGoaA2CzZ7kqA?usp=drive_link
- Download images for OSIE from https://github.com/NUS-VIP/predicting-human-gaze-beyond-pixels/tree/master/data/stimuli
- Download labels from [https://drive.google.com/drive/folders/11TyynVjKkHWw84WDZwEgOxZBuUdmxhlT?usp=drive_link](https://drive.google.com/drive/folders/11TyynVjKkHWw84WDZwEgOxZBuUdmxhlT?usp=drive_link)
- Put it under data folder.
- Download checkpoints and subject embeddings from https://drive.google.com/drive/folders/12NWP6ETNS7IRfhXOqCyqSrsHMXyLNiAG?usp=drive_link

#### Generate unseen subject embeddings from checkpoints
- OSIE
    ```
    CUDA_VISIBLE_DEVICES=0 python train.py \
    --hparams ./configs/osie_useremb.json \
    --dataset-root data \
    --fewshot_subject 10 11 12 13 14 \
    --eval-only
    ```
- COCO_FV
    ```
    CUDA_VISIBLE_DEVICES=0 python train.py \
    --hparams ./configs/coco_freeview_useremb.json \
    --dataset-root data \
    --fewshot_subject 0 1 2 \
    --eval-only
    ```
- COCO_Search18
    ```
    CUDA_VISIBLE_DEVICES=0 python train.py \
    --hparams ./configs/coco_search18_TP_useremb.json \
    --dataset-root data \
    --fewshot_subject 7 8 9 \
    --eval-only
    ```
    
#### Train Network
- Example: for OSIE, take the first 10 subjects as seen, and last 5 subjects (10, 11, 12, 13, 14) as unseen.
- Train SE-Net with
    ```
    CUDA_VISIBLE_DEVICES=0 python train.py \
    --hparams ./configs/osie_useremb.json \
    --dataset-root data \
    --ex_subject 10 11 12 13 14
    ```
- Generate seen subject embeddings with
    ```
    CUDA_VISIBLE_DEVICES=5 python train.py \
    --hparams ./configs/osie_useremb.json \
    --dataset-root data \
    --ex_subject 10 11 12 13 14 \
    --eval-only
    ```
- Generate unseen subject embeddings with
    ```
    CUDA_VISIBLE_DEVICES=0 python train.py \
    --hparams ./configs/osie_useremb.json \
    --dataset-root data \
    --fewshot_subject 10 11 12 13 14 \
    --eval-only
    ```
- Evaluate classification accuracy with
    ```
    CUDA_VISIBLE_DEVICES=5 python train.py \
    --hparams ./configs/osie_useremb.json \
    --dataset-root data \
    --ex_subject 10 11 12 13 14 \
    --eval-only \
    --mode evaluate-net
    ```

#### Acknowledgement
Code of SE-Net is built upon HAT[1]. 

[1] Zhibo Yang, Sounak Mondal, Seoyoung Ahn, Ruoyu Xue, Gregory Zelinsky, Minh Hoai, and Dimitris Samaras. Unifying top-down and bottom-up scanpath prediction using transformers. CVPR 2024.


