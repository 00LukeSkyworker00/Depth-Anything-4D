export CUDA_VISIBLE_DEVICES=0,1,4,6,7
python train.py \
    --data-dir /home/skyworker/data/real_estate_10k/DFoT \
    --out-dir /home/skyworker/result/da3_4d \
    --batch 4 \
    --ep-len 3 \
    --epoch 50 \
    --max-lr 0.00005 \