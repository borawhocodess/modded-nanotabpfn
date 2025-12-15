export OMP_NUM_THREADS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

torchrun --nproc_per_node=8 --master-port=29501 train_nano.py
