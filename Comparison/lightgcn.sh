DATASET='epinions'

python lightgcn.py \
    --dataset $DATASET \
    --device 0 \
    --model "lightgcn" \
    --dataset_type "social" \
    --ori_lr 0.001 \
    --aug_lr 0.001 \
    --batch_size 4096 \
    --epochs 105 \
    --threshold 100 \
    --n_layer 2 \
    --add_edge 0 \
    --load_partition 1 \
    --eval_steps 1 \
    --aug_size 1.25 \
    --alpha 0.002 \
    --beta 100 \
    --aug_type 'rw' \
    --train_ratio 1 \



DATASET='dblp'

python lightgcn.py \
    --dataset $DATASET \
    --dataset_type "social" \
    --device 1 \
    --model "lightgcn" \
    --ori_lr 0.001 \
    --aug_lr 0.001 \
    --batch_size 4096 \
    --epochs 105 \
    --threshold 25 \
    --n_layer 1 \
    --add_edge 0 \
    --load_partition 1 \
    --eval_steps 1 \
    --aug_size 1 \
    --aug_type 'rw' \
    --train_ratio 1 \
    --alpha 0.0005 \
    --beta 20 \