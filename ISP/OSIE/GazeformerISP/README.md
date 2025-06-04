#### ISP-SENet (Predict personalized scanpaths)
 - Refer to ISP[2] https://github.com/chenxy99/IndividualScanpath for installation.
 - Generate image_features using preprocess/feature_extractor.py and store in data folder.

#### Run DEMO
 - test scanpath prediction performance on query set (unseen subjects).
    ```
    CUDA_VISIBLE_DEVICES=0 python src/test.py --fewshot_subject 10 11 12 13 14
    ```
 - test scanpath prediction performance on base set (seen subjects).
    ```
    CUDA_VISIBLE_DEVICES=0 python src/test.py --ex_subject 10 11 12 13 14 ----user_emb_path assets/isp/osie-ex-10to15/train_user_embedding.pt
    ```

#### Train Network
 - Train ISP on base set with
    ```
    sh bash/train.sh
    ```
  - Test ISP on query set with
    ```
    CUDA_VISIBLE_DEVICES=0 python src/test.py
    ```

#### Acknowledgement
Code is modified from ISP[2]

[2] Xianyu Chen, Ming Jiang, and Qi Zhao. Beyond average: Individualized visual scanpath prediction. CVPR 2024.

