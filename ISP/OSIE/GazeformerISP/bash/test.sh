for i in $(seq 0 10)
do
  CUDA_VISIBLE_DEVICES=3 python src/test.py \
  --evaluation_dir assets/OSIE-ex-10to15-$i \
  --cuda 1 \
  --random_support $i
done
