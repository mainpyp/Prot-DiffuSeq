# generation
conda activate henkel_diffuseq

python -u run_decode.py \
--model_dir /mnt/project/henkel/repositories/Prot-DiffuSeq/diffusion_models/diffuseq_ProtMediumCorrect_h256_lr1e-05_t6000_sqrt_lossaware_seed123_pm-correct-new-params20230419-17:39:32/ \
--n_gpus 3 \
--seeds 100 120 420 12 49 12 \
--generate 

conda deactivate

# analyze generation
conda activate mheinzinger_esmfold_CLI

python analyze_generation.py \
-i /mnt/project/henkel/repositories/Prot-DiffuSeq/generation_outputs/diffuseq_ProtMediumCorrect_h256_lr1e-05_t6000_sqrt_lossaware_seed123_pm-correct-new-params20230419-17:39:32/

conda deactivate