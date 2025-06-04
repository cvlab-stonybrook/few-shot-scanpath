for i in $(seq 0 5)
do
  CUDA_VISIBLE_DEVICES=3 python src/test.py \
  --evaluation_dir /nfs/130.245.4.102/add_disk4/ruoyu/project/ISP/assets/osie-original-ex-01234-only-subemb-fewshot-10-randommmmm$i \
  --cuda 1 \
  --random_support $i
done
