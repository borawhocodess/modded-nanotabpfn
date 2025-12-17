export OMP_NUM_THREADS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

export TORCH_SHOW_CPP_STACKTRACES=1
export NCCL_DEBUG=INFO

torchrun --nproc_per_node=8 --master-port=29501 train_nano.py
