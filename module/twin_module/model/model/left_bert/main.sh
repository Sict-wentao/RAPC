python main.py \
    --train_dataset "/home/zhangwentao/host/project/paper/data/pretrain_data/wiki_pire/train.txt" \
    --test_dataset  "/home/zhangwentao/host/project/paper/data/pretrain_data/wiki_pire/dev.txt" \
    --vocab_path    "/home/zhangwentao/host/project/paper/model/model/left_bert/dataset/vocab.pkl" \
    --output_path   "/home/zhangwentao/host/project/paper/model/model/left_bert/output/run-0531" \
    --on_memory     False \
    --lr 1e-4 \
    --cuda_devices  0 \
    --hidden        768 \
    --layers        12 \
    --attn_heads    12 \
    --seq_len       128 \
    --batch_size    128 \
    --save_step     1000 \
    --epochs        20 \
    --num_workers   8 \
    