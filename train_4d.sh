export CUDA_VISIBLE_DEVICES=0
python train.py \
    --data-dir /home/skyworker/data/real_estate_10k/DFoT \
    --out-dir /home/skyworker/result/da3_4d \
    --batch 4 \
    --ep-len 3 \