#### Installation
 - You should create seperate env for SE-Net and ISP
 - install required packages by:
   ```
   conda env create -f environment.yml
   ```
   
 - For other questions of setup, refer to ISP[2] for environment setup: https://github.com/chenxy99/IndividualScanpath/tree/main.

#### Data Prepration
 - Run src/preprocess/feature_extractor.py to get image features, and store under the data/image_features folder.

#### Checkpoint
 - put checkpoint_best.pt in <dataset_name>/GazeformerISP/src/assets/<log_dir>/checkpoints/checkpoint_best.pt

#### Training and Inference
 - Please go to the corresponding folders of different datasets for implementation details.

#### Acknowledgement
 - Code is modified from ISP[2]
