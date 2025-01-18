#!/bin/bash


if [ "$EUID" -ne 0 ]; then 
    echo "Run as root you shmuk"
    exit 1
fi

# TensorFlow memory settings
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPU_ALLOCATOR_CHUNK_SIZE=1

# Memory allocation limits
export MALLOC_ARENA_MAX=4
export MALLOC_MMAP_THRESHOLD_=131072
export MALLOC_TRIM_THRESHOLD_=131072
export MALLOC_TOP_PAD_=131072

# Enable memory overcommit
sysctl -w vm.overcommit_memory=1

# Set matplotlib backend
export MPLBACKEND=TkAgg

# Make environment variables available to child processes
printenv | grep -E "^(TF_|MALLOC_|MPLBACKEND=)" >> /etc/radosgqcnn

echo "Environment variables for Rados wsl cpu setup executed"
echo "Now source"
echo "source /etc/radosgqcnn"
echo "also first:"
echo "source gqVenv/bin/activate"