accelerate launch run_train.py \
	--diff_steps 3000 \
	--lr 0.00001 \
	--learning_steps 100001 \
	--save_interval 500 \
	--seed 123 \
	--noise_schedule sqrt \
	--hidden_dim 1024 \
	--bsz 1024 \
	--dataset ProtMediumLRZ \
	--data_dir datasets/ProtMediumLRZ \
	--vocab protbert \
	--seq_len 256 \
	--schedule_sampler lossaware \
	--notes pm-medium-generalprobe
