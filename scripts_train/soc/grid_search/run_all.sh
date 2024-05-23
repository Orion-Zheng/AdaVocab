# Test all the hyperparameters for AdaVocab on SoC dataset
# Learning Rate: 2e-4, 2e-5, 2e-6 (different scales)
# Warmup Ratio: 0.03, 200 (different settings)
bash scripts_train/soc/grid_search/AdaVocab_2_epoch_topk_800_run_dist_train_h100_lr_2e-4.sh
bash scripts_train/soc/grid_search/AdaVocab_2_epoch_topk_800_run_dist_train_h100_lr_2e-5_wr_200.sh
bash scripts_train/soc/grid_search/AdaVocab_2_epoch_topk_800_run_dist_train_h100_lr_2e-5.sh
bash scripts_train/soc/grid_search/AdaVocab_2_epoch_topk_800_run_dist_train_h100_lr_2e-6.sh