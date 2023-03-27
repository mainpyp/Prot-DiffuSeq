python -m torch.distributed.launch --nproc_per_node=2 --master_port=12233 --use_env run_train.py \
--diff_steps 4000 \
--lr 0.0001 \
--learning_steps 5001 \
--save_interval 1000 \
--seed 102 \
--noise_schedule sqrt \
--hidden_dim 128 \
--bsz 256 \
--dataset ProtMedium \
--data_dir datasets/ProtMediumCorrect \
--vocab protbert \
--seq_len 128 \
--schedule_sampler lossaware \
--notes test-ccs
