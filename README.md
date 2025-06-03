# few-shot-scanpath

CVPR 2025 "Few-shot Personalized Scanpath Prediction"

#### Installation
 - For SE-Net, use the installation in https://github.com/cvlab-stonybrook/HAT?tab=readme-ov-file:
 - 1) Install [Detectron2](https://github.com/facebookresearch/detectron2)
 - 2) Install MSDeformableAttn:
   ```
   cd ./hat/pixel_decoder/ops
   sh make.sh
   ```
#### Data
- Download all required data for SE-Net from https://drive.google.com/drive/folders/12UZn-kRvvaGR5Qbhkg6xMTSl_SK7LNM_?usp=drive_link
- Put it under data folder.
 
#### SE-Net
- Train SE-Net with
    ```
    CUDA_VISIBLE_DEVICES=0 python train.py --hparams ./configs/osie_useremb.json --dataset-root data
    ```
- Generate seen subject embeddings with
    ```
    CUDA_VISIBLE_DEVICES=0 python train.py --hparams ./configs/osie_useremb.json --dataset-root data --eval-only
    ```
- Generate unseen subject embeddings with
    ```
    CUDA_VISIBLE_DEVICES=0 python train.py --hparams ./configs/osie_useremb.json --dataset-root data --eval-only
    ```

#### ISP-SENet (Predict personalized scanpaths)
 - Refer to https://github.com/chenxy99/IndividualScanpath for installation and data prepration.
 - Train ISP on base set with
    ```
    sh bash/train.sh
    ```
  - Test ISP on query set with
    ```
    python src/test.py
    ```

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
