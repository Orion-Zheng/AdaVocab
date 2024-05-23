# Test all the hyperparameters for AdaVocab on SoC dataset
# Learning Rate: 2e-4, 2e-5(different scales)  --> 2e-6 is too low
# Warmup Ratio: 0.03(~50), 0.05(~100) (different settings)
bash scripts_train/soc/grid_search/tinyllama_dist_train_h100_topk_800_lr_2e-4_warm_50_default.sh
bash scripts_train/soc/grid_search/tinyllama_dist_train_h100_topk_800_lr_2e-4_warm_50_no_topk_loss.sh
bash scripts_train/soc/grid_search/tinyllama_dist_train_h100_topk_800_lr_2e-4_warm_50_r_8.sh
bash scripts_train/soc/grid_search/tinyllama_dist_train_h100_topk_800_lr_2e-4_warm_50_r_16.sh
# warmup steps
bash scripts_train/soc/grid_search/tinyllama_dist_train_h100_topk_800_lr_2e-4_warm_100_default.sh
# learning rate
bash scripts_train/soc/grid_search/tinyllama_dist_train_h100_topk_800_lr_2e-5_warm_50_default.sh