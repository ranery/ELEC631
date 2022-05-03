CUDA_VISIBLE_DEVICES=8 python train_model.py

CUDA_VISIBLE_DEVICES=7 python train_model.py \
problem.hyp.alpha=0.01 \
problem/model=dt_net_recall_2d \
problem=mazes \
name=mazes_ablation

CUDA_VISIBLE_DEVICES=6 python train_model.py \
problem.hyp.alpha=0.01 \
problem/model=dt_net_recall_2d_reinit \
problem=mazes \
name=mazes_reinit_ablation

CUDA_VISIBLE_DEVICES=9 python train_model.py \
problem.hyp.alpha=0.1 \
problem.hyp.train_batch_size=200 \
problem.hyp.test_batch_size=200 \
problem/model=dt_net_recall_2d \
problem=chess \
name=chess_ablation


# test prefix

CUDA_VISIBLE_DEVICES=6 python test_model.py \
problem.model.model_path=/home2/hy34/ELEC631/deep-thinking/outputs/training_default/training-nary-Kimberlie \
problem.model.test_iterations.low=30 \
problem.model.test_iterations.high=600 \
name=prefix_test_best \
quick_test=True

python deepthinking/data_analysis/make_schoop.py \
outputs/prefix_test_best

# test maze
CUDA_VISIBLE_DEVICES=9 python test_model.py \
problem.model.model_path=/home2/hy34/ELEC631/deep-thinking/outputs/mazes_ablation/training-wageless-Greta \
problem.hyp.alpha=0.01 \
problem/model=dt_net_recall_2d \
problem=mazes \
problem.model.test_iterations.low=30 \
problem.model.test_iterations.high=33 \
name=mazes_ablation_test \
quick_test=True

CUDA_VISIBLE_DEVICES=9 python test_model.py \
problem.model.model_path=/home2/hy34/ELEC631/deep-thinking/outputs/mazes_ablation/training-wageless-Greta \
problem.hyp.alpha=0.01 \
problem/model=dt_net_recall_2d \
problem=mazes \
problem.model.test_iterations.low=30 \
problem.model.test_iterations.high=50 \
name=mazes_ablation_test \
quick_test=True

CUDA_VISIBLE_DEVICES=8 python test_model.py \
problem.model.model_path=/home2/hy34/ELEC631/deep-thinking/outputs/mazes_ablation/training-wageless-Greta \
problem.hyp.alpha=0.01 \
problem/model=dt_net_recall_2d \
problem=mazes \
problem.model.test_iterations.low=50 \
problem.model.test_iterations.high=100 \
name=mazes_ablation_test \
quick_test=True

CUDA_VISIBLE_DEVICES=7 python test_model.py \
problem.model.model_path=/home2/hy34/ELEC631/deep-thinking/outputs/mazes_ablation/training-wageless-Greta \
problem.hyp.alpha=0.01 \
problem/model=dt_net_recall_2d \
problem=mazes \
problem.model.test_iterations.low=100 \
problem.model.test_iterations.high=200 \
name=mazes_ablation_test \
quick_test=True

CUDA_VISIBLE_DEVICES=8 python test_model.py \
problem.model.model_path=/home2/hy34/ELEC631/deep-thinking/outputs/mazes_ablation/training-wageless-Greta \
problem.hyp.alpha=0.01 \
problem/model=dt_net_recall_2d \
problem=mazes \
problem.model.test_iterations.low=200 \
problem.model.test_iterations.high=300 \
name=mazes_ablation_test \
quick_test=True

CUDA_VISIBLE_DEVICES=7 python test_model.py \
problem.model.model_path=/home2/hy34/ELEC631/deep-thinking/outputs/mazes_ablation/training-wageless-Greta \
problem.hyp.alpha=0.01 \
problem/model=dt_net_recall_2d \
problem=mazes \
problem.model.test_iterations.low=300 \
problem.model.test_iterations.high=1000 \
name=mazes_ablation_test \
quick_test=True

CUDA_VISIBLE_DEVICES=9 python test_model.py \
problem.model.model_path=/home2/hy34/ELEC631/deep-thinking/outputs/mazes_ablation/training-wageless-Greta \
problem.hyp.alpha=0.01 \
problem/model=dt_net_recall_2d \
problem=mazes \
problem.model.test_iterations.low=30 \
problem.model.test_iterations.high=1000 \
name=mazes_ablation_test_best \
quick_test=True

python deepthinking/data_analysis/make_schoop.py \
outputs/mazes_ablation_test_best


## Train pruned models (prefix sums 30 --> 512)
CUDA_VISIBLE_DEVICES=3 python train_model.py \
problem.hyp.alpha=1 \
problem/model=dt_net_recall_1d_pruned \
problem=prefix_sums \
name=prefix_sums_pruned_0.6 \
prune.ifprune=True \
prune.ratio=0.6 \
prune.method=rand

CUDA_VISIBLE_DEVICES=5 python train_model.py \
problem.hyp.alpha=1 \
problem/model=dt_net_recall_1d_pruned \
problem=prefix_sums \
name=prefix_sums_pruned_mag_0.6 \
prune.ifprune=True \
prune.ratio=0.6 \
prune.method=mag

CUDA_VISIBLE_DEVICES=9 python train_model.py \
problem.hyp.alpha=1 \
problem/model=dt_net_recall_1d_pruned \
problem=prefix_sums \
name=prefix_sums_pruned_snip_0.3_new \
prune.ifprune=True \
prune.ratio=0.3 \
prune.method=snip

CUDA_VISIBLE_DEVICES=6 python train_model.py \
problem.hyp.alpha=1 \
problem/model=dt_net_recall_1d_pruned \
problem=prefix_sums \
name=prefix_sums_pruned_grasp_0.6 \
prune.ifprune=True \
prune.ratio=0.6 \
prune.method=grasp

CUDA_VISIBLE_DEVICES=3 python train_model.py \
problem.hyp.alpha=1 \
problem/model=dt_net_recall_1d_pruned \
problem=prefix_sums \
name=prefix_sums_pruned_synflow_0.9 \
prune.ifprune=True \
prune.ratio=0.9 \
prune.method=synflow

## Test pruned models
CUDA_VISIBLE_DEVICES=5 python test_model.py \
problem.hyp.alpha=1 \
problem/model=dt_net_recall_1d_pruned \
problem=prefix_sums \
prune.ifprune=True \
prune.ratio=0.6 \
prune.method=rand \
problem.model.model_path=/home2/hy34/ELEC631/deep-thinking/outputs/prefix_sums_pruned_0.6/training-spleenful-Tiwana \
problem.model.test_iterations.low=30 \
problem.model.test_iterations.high=600 \
name=prefix_test_rand_0.6 \
quick_test=True

CUDA_VISIBLE_DEVICES=6 python test_model.py \
problem.hyp.alpha=1 \
problem/model=dt_net_recall_1d_pruned \
problem=prefix_sums \
prune.ifprune=True \
prune.ratio=0.6 \
prune.method=mag \
problem.model.model_path=/home2/hy34/ELEC631/deep-thinking/outputs/prefix_sums_pruned_mag_0.6/training-hatless-Jacqueleen \
problem.model.test_iterations.low=30 \
problem.model.test_iterations.high=600 \
name=prefix_test_mag_0.6 \
quick_test=True

CUDA_VISIBLE_DEVICES=7 python test_model.py \
problem.hyp.alpha=1 \
problem/model=dt_net_recall_1d_pruned \
problem=prefix_sums \
prune.ifprune=True \
prune.ratio=0.6 \
prune.method=snip \
problem.model.model_path=/home2/hy34/ELEC631/deep-thinking/outputs/prefix_sums_pruned_snip_0.6/training-dateless-Chivas \
problem.model.test_iterations.low=30 \
problem.model.test_iterations.high=600 \
name=prefix_test_snip_0.6 \
quick_test=True

CUDA_VISIBLE_DEVICES=8 python test_model.py \
problem.hyp.alpha=1 \
problem/model=dt_net_recall_1d_pruned \
problem=prefix_sums \
prune.ifprune=True \
prune.ratio=0.3 \
prune.method=snip \
problem.model.model_path=/home2/hy34/ELEC631/deep-thinking/outputs/prefix_sums_pruned_snip_0.3_new/training-flimsy-Latissa \
problem.model.test_iterations.low=30 \
problem.model.test_iterations.high=600 \
name=prefix_test_snip_0.3_new \
quick_test=True

CUDA_VISIBLE_DEVICES=4 python test_model.py \
problem.hyp.alpha=1 \
problem/model=dt_net_recall_1d_pruned \
problem=prefix_sums \
prune.ifprune=True \
prune.ratio=0.6 \
prune.method=grasp \
problem.model.model_path=/home2/hy34/ELEC631/deep-thinking/outputs/prefix_sums_pruned_grasp_0.9/training-truer-Shey \
problem.model.test_iterations.low=30 \
problem.model.test_iterations.high=600 \
name=prefix_test_grasp_0.6 \
quick_test=True


## draw
python deepthinking/data_analysis/make_schoop.py \
outputs/prefix_test_rand_0.1

python deepthinking/data_analysis/my_make_schoop.py \
--filepath_unpruned 'outputs/prefix_test_best' \
--filepath_rand 'outputs/prefix_test_rand_0.1' \
--filepath_mag 'outputs/prefix_test_mag_0.1' \
--filepath_snip 'outputs/prefix_test_snip_0.1' \
--filepath_grasp 'outputs/prefix_test_grasp_0.1' \
--plot_name 'prefix_sums_0.1.png' \
--plot_title 'Prefix Sums 30 to 512 (p = 10%)' \
--xlim 0 600

python deepthinking/data_analysis/my_make_schoop.py \
--filepath_unpruned 'outputs/prefix_test_best' \
--filepath_rand 'outputs/prefix_test_rand_0.3' \
--filepath_mag 'outputs/prefix_test_mag_0.3' \
--filepath_snip 'outputs/prefix_test_snip_0.3' \
--plot_name 'prefix_sums_0.3.png' \
--plot_title 'Prefix Sums 30 to 512 (p = 30%)' \
--xlim 0 600

python deepthinking/data_analysis/my_make_schoop.py \
--filepath_unpruned 'outputs/prefix_test_best' \
--filepath_rand 'outputs/prefix_test_rand_0.6' \
--filepath_mag 'outputs/prefix_test_mag_0.6' \
--filepath_snip 'outputs/prefix_test_snip_0.6' \
--plot_name 'prefix_sums_0.6.png' \
--plot_title 'Prefix Sums 30 to 512 (p = 60%)' \
--xlim 0 600

python deepthinking/data_analysis/my_make_schoop.py \
--filepath_unpruned 'outputs/prefix_test_best' \
--filepath_rand 'outputs/prefix_test_rand_0.9' \
--filepath_mag 'outputs/prefix_test_mag_0.9' \
--filepath_snip 'outputs/prefix_test_snip_0.9' \
--filepath_grasp 'outputs/prefix_test_grasp_0.9' \
--plot_name 'prefix_sums_0.9.png' \
--plot_title 'Prefix Sums 30 to 512 (p = 90%)' \
--xlim 0 600


## Train pruned models (maze 9x9 --> 13x13)
# 2/7/8

CUDA_VISIBLE_DEVICES=2 python train_model.py \
problem.hyp.alpha=0.01 \
problem/model=dt_net_recall_2d \
problem=mazes \
name=mazes_unpruned

CUDA_VISIBLE_DEVICES=7 python train_model.py \
problem.hyp.alpha=0.01 \
problem/model=dt_net_recall_2d_pruned \
problem=mazes \
name=mazes_pruned_rand_0.1 \
prune.ifprune=True \
prune.ratio=0.1 \
prune.method=rand

CUDA_VISIBLE_DEVICES=2 python train_model.py \
problem.hyp.alpha=0.01 \
problem/model=dt_net_recall_2d_pruned \
problem=mazes \
name=mazes_pruned_mag_0.1 \
prune.ifprune=True \
prune.ratio=0.1 \
prune.method=mag

CUDA_VISIBLE_DEVICES=8 python train_model.py \
problem.hyp.alpha=0.01 \
problem/model=dt_net_recall_2d_pruned \
problem=mazes \
name=mazes_pruned_snip_0.1 \
prune.ifprune=True \
prune.ratio=0.1 \
prune.method=snip

CUDA_VISIBLE_DEVICES=0 python train_model.py \
problem.hyp.alpha=0.01 \
problem/model=dt_net_recall_2d_pruned \
problem=mazes \
name=mazes_pruned_snip_0.1_update \
prune.ifprune=True \
prune.ratio=0.1 \
prune.method=snip

CUDA_VISIBLE_DEVICES=1 python train_model.py \
problem.hyp.alpha=0.01 \
problem/model=dt_net_recall_2d_pruned \
problem=mazes \
name=mazes_pruned_grasp_0.1 \
prune.ifprune=True \
prune.ratio=0.1 \
prune.method=grasp

## Test pruned models
CUDA_VISIBLE_DEVICES=3 python test_model.py \
problem.hyp.alpha=0.01 \
problem/model=dt_net_recall_2d_pruned \
problem=mazes \
problem.model.model_path=/home2/hy34/ELEC631/deep-thinking/outputs/mazes_unpruned/training-tactless-Gamaliel \
problem.model.test_iterations.low=30 \
problem.model.test_iterations.high=1000 \
name=mazes_unpruned_test \
quick_test=True


CUDA_VISIBLE_DEVICES=3 python test_model.py \
problem.hyp.alpha=0.01 \
problem/model=dt_net_recall_2d_pruned \
problem=mazes \
prune.ifprune=True \
prune.ratio=0.1 \
prune.method=rand \
problem.model.model_path=/home2/hy34/ELEC631/deep-thinking/outputs/mazes_pruned_rand_0.1/training-tandem-Odelia \
problem.model.test_iterations.low=30 \
problem.model.test_iterations.high=1000 \
name=mazes_rand_0.1 \
quick_test=True

CUDA_VISIBLE_DEVICES=4 python test_model.py \
problem.hyp.alpha=0.01 \
problem/model=dt_net_recall_2d_pruned \
problem=mazes \
prune.ifprune=True \
prune.ratio=0.1 \
prune.method=mag \
problem.model.model_path=/home2/hy34/ELEC631/deep-thinking/outputs/mazes_pruned_mag_0.1/training-doggoned-Siddharth \
problem.model.test_iterations.low=30 \
problem.model.test_iterations.high=1000 \
name=mazes_mag_0.1 \
quick_test=True

CUDA_VISIBLE_DEVICES=4 python test_model.py \
problem.hyp.alpha=0.01 \
problem/model=dt_net_recall_2d_pruned \
problem=mazes \
prune.ifprune=True \
prune.ratio=0.1 \
prune.method=snip \
problem.model.model_path=/home2/hy34/ELEC631/deep-thinking/outputs/mazes_pruned_snip_0.1/training-petrous-Lezette \
problem.model.test_iterations.low=30 \
problem.model.test_iterations.high=1000 \
name=mazes_snip_0.1 \
quick_test=True

# vis
python deepthinking/data_analysis/my_make_schoop.py \
--filepath_unpruned 'outputs/mazes_unpruned_test' \
--filepath_rand 'outputs/mazes_rand_0.1' \
--filepath_snip 'outputs/mazes_snip_0.1' \
--plot_name 'maze_0.1.png' \
--plot_title 'Maze 9x9 to 13x13 (p = 10%)' \
--xlim 0 600 \
--ylim 30 100