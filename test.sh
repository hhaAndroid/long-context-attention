export PYTHONPATH="$(pwd)"
echo 'test/test_ring_flash_attn_func.py'
srun -p llm_razor -N 1 --gres=gpu:4 --ntasks=1 --cpus-per-task=16 --time=1:00:00 torchrun --nproc-per-node=4 test/test_ring_flash_attn_func.py
echo 'test/test_ring_flash_attn_varlen_func.py'
srun -p llm_razor -N 1 --gres=gpu:4 --ntasks=1 --cpus-per-task=16 --time=1:00:00 torchrun --nproc-per-node=4 test/test_ring_flash_attn_varlen_func.py
echo 'test/test_zigzag_ring_flash_attn_func.py'
srun -p llm_razor -N 1 --gres=gpu:4 --ntasks=1 --cpus-per-task=16 --time=1:00:00 torchrun --nproc-per-node=4 test/test_zigzag_ring_flash_attn_func.py
echo 'test/test_zigzag_ring_flash_attn_varlen_func.py'
srun -p llm_razor -N 1 --gres=gpu:4 --ntasks=1 --cpus-per-task=16 --time=1:00:00 torchrun --nproc-per-node=4 test/test_zigzag_ring_flash_attn_varlen_func.py
echo 'test/test_hybrid_attn.py'
srun -p llm_razor -N 1 --gres=gpu:4 --ntasks=1 --cpus-per-task=16 --time=1:00:00 torchrun --nproc-per-node=4 test/test_hybrid_attn.py
echo 'test/test_hybrid_qkvpacked_attn.py'
srun -p llm_razor -N 1 --gres=gpu:4 --ntasks=1 --cpus-per-task=16 --time=1:00:00 torchrun --nproc-per-node=4 test/test_hybrid_qkvpacked_attn.py
echo 'test/test_llama3_flash_attn_varlen_func.py'
srun -p llm_razor -N 1 --gres=gpu:4 --ntasks=1 --cpus-per-task=16 --time=1:00:00 torchrun --nproc-per-node=4 test/test_llama3_flash_attn_varlen_func.py
