DATASET='dblp'

python improve.py \
    --dataset $DATASET \
    --device 0 \
    --model "lightgcn" \
    --dataset_type "social" \
    --ori_lr 0.001 \
    --aug_lr 0.001 \
    --batch_size 4096 \
    --epochs 300 \
    --threshold 100 \
    --n_layer 2 \
    --add_edge 0 \
    --load_partition 1 \
    --eval_steps 1 \
    --aug_size 1.25 \
    --alpha 0.002 \
    --beta 100 \
    --aug_type 'jaccard' \
    --train_ratio 1 \

