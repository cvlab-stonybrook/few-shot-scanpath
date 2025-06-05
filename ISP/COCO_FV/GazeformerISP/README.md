
#### Run DEMO
 - test scanpath prediction performance on query set (unseen subjects).
    ```
    CUDA_VISIBLE_DEVICES=0 python src/test.py --fewshot_subject 0 1 2
    ```
 - test scanpath prediction performance on base set (seen subjects).
    ```
    CUDA_VISIBLE_DEVICES=0 python src/test.py --ex_subject 0 1 2 --user_emb_path ../../../SE-Net/assets/FV-ex-012/train_user_embedding.pt
    ```

#### Train Network
 - Train ISP on base set with
    ```
    sh bash/train.sh
    ```
  - Test ISP on query set with
    ```
    CUDA_VISIBLE_DEVICES=0 python src/test.py --fewshot_subject 0 1 2
    ```

#### Acknowledgement
Code is modified from ISP[2]

[2] Xianyu Chen, Ming Jiang, and Qi Zhao. Beyond average: Individualized visual scanpath prediction. CVPR 2024.

