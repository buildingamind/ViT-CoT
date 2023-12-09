viewpoint=V1O1

for viewpoint in V1O1
do
    python3 train_vit.py \
        --max_epochs 100 \
        --batch_size 128 \
        --data_dir /data/lpandey/paper1/${viewpoint} \
       	--seed_val 0 \
        --temporal \
        --shuffle False \
        --window_size 3 \
        --head 1 \
        --temporal_mode 2+images \
        --drop_ep 0 \
        --val_split 0.05 --exp_name dummy/${viewpoint}
done
