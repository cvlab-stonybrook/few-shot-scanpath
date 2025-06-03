# few-shot-scanpath

CVPR 2025 "Few-shot Personalized Scanpath Prediction"

#### Installation
 - For SE-Net, use the installation in HAT[1] https://github.com/cvlab-stonybrook/HAT?tab=readme-ov-file:
 - 1) Install [Detectron2](https://github.com/facebookresearch/detectron2)
 - 2) Install MSDeformableAttn:
   ```
   cd ./hat/pixel_decoder/ops
   sh make.sh
   ```
#### Data
- Download images for COCO-Search18 and COCO-FreeView from https://drive.google.com/drive/folders/1im5SJKQ976MmB7JeRgtMGoaA2CzZ7kqA?usp=drive_link
- Download images for OSIE from https://github.com/NUS-VIP/predicting-human-gaze-beyond-pixels/tree/master/data/stimuli
- Download labels from https://drive.google.com/drive/folders/12UZn-kRvvaGR5Qbhkg6xMTSl_SK7LNM_?usp=drive_link
- Put it under data folder.
 
#### SE-Net (Generate subject embeddings)
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
    
#### ISP-SENet (Predict personalized scanpaths)
 - Refer to ISP[2] https://github.com/chenxy99/IndividualScanpath for installation and data prepration.
 - Train ISP on base set with
    ```
    sh bash/train.sh
    ```
  - Test ISP on query set with
    ```
    python src/test.py
    ```

#### Acknowledgement
Code of SE-Net is built upon HAT[1]. 

[1] Zhibo Yang, Sounak Mondal, Seoyoung Ahn, Ruoyu Xue, Gregory Zelinsky, Minh Hoai, and Dimitris Samaras. Unifying top-down and bottom-up scanpath prediction using transformers. CVPR 2024.

[2] Xianyu Chen, Ming Jiang, and Qi Zhao. Beyond average: Individualized visual scanpath prediction. CVPR 2024.
####Reference
Please cite if you use this code base.

```bibtex
@article{xue2025few,
  title={Few-shot Personalized Scanpath Prediction},
  author={Xue, Ruoyu and Xu, Jingyi and Mondal, Sounak and Le, Hieu and Zelinsky, Gregory and Hoai, Minh and Samaras, Dimitris},
  journal={arXiv preprint arXiv:2504.05499},
  year={2025}
}

```
