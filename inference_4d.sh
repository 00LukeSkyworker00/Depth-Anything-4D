export CUDA_VISIBLE_DEVICES=7
python inference_4d.py \
    --data-dir /home/skyworker/data/real_estate_10k/DFoT \
    --out-dir /home/skyworker/result/da3_4d/2026-04-03-BEST \
    --ckpt-dir /home/skyworker/result/da3_4d/2026-04-03-BEST \
    --batch 1 \
    --ep-len 3 \
