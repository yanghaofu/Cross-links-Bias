DATASET='epinions'

python e2e_main.py \
    --dataset $DATASET \
    --device 0 \
    --model "gat" \
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
    --alpha 0.00005 \
    --beta 100 \
    --aug_type 'rw' \
    --train_ratio 1 \
    --dataset_type 'social' \


DATASET='epinions'

python e2e_main.py \
    --dataset $DATASET \
    --device 0 \
    --model "gin" \
    --ori_lr 0.001 \
    --aug_lr 0.001 \
    --batch_size 4096 \
    --epochs 300 \
    --threshold 100 \
    --n_layer 1 \
    --add_edge 0 \
    --load_partition 1 \
    --eval_steps 1 \
    --aug_size 1 \
    --alpha 0.00005 \
    --beta 100 \
    --aug_type 'rw' \
    --train_ratio 1 \
    --dataset_type 'social' \


DATASET='epinions'

python e2e_main.py \
    --dataset $DATASET \
    --device 0 \
    --model "sage" \
    --ori_lr 0.001 \
    --aug_lr 0.001 \
    --batch_size 4096 \
    --epochs 300 \
    --threshold 50 \
    --n_layer 1 \
    --add_edge 0 \
    --load_partition 1 \
    --eval_steps 1 \
    --aug_size 1.25 \
    --alpha 0.00005 \
    --beta 100 \
    --aug_type 'rw' \
    --train_ratio 1 \
    --dataset_type 'social' \


DATASET='epinions'

python e2e_main.py \
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
    --aug_type 'rw' \
    --train_ratio 1 \


DATASET='epinions'

python e2e_main.py \
    --dataset $DATASET \
    --device 0 \
    --model "ultragcn" \
    --dataset_type "social" \
    --ori_lr 0.005 \
    --aug_lr 0.005 \
    --batch_size 4096 \
    --epochs 300 \
    --threshold 50 \
    --n_layer 2 \
    --add_edge 0 \
    --load_partition 1 \
    --eval_steps 1 \
    --aug_size 1 \
    --alpha 0.00005 \
    --beta 100 \
    --train_ratio 1 \
    --aug_type 'rw' \


DATASET='epinions'

python e2e_main.py \
    --dataset $DATASET \
    --device 0 \
    --model "pprgo" \
    --dataset_type "social" \
    --ori_lr 0.004 \
    --aug_lr 0.004 \
    --batch_size 4096 \
    --epochs 300 \
    --threshold 100 \
    --n_layer 2 \
    --add_edge 0 \
    --load_partition 1 \
    --eval_steps 1 \
    --aug_size 1.25 \
    --alpha 0.00005 \
    --beta 100 \
    --aug_type 'rw' \
