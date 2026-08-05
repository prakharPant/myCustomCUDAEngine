import torch
from typing import List, Dict

class KVCacheManager:
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda"
    ):
        self.block_size = block_size
        self.num_blocks = num_blocks
        
        # Physical KV Cache Tensor Allocation
        # [num_layers, 2, num_blocks, block_size, num_heads, head_dim]
        # (2 corresponds to Key and Value)
        self.kv_cache = torch.empty(
            (num_layers, 2, num_blocks, block_size, num_heads, head_dim),
            dtype=dtype,
            device=device
        )
        
        # Free list of physical block indices
        self.free_blocks = list(range(num_blocks))
        # Logical sequence -> List of physical block IDs
        self.block_tables: Dict[int, List[int]] = {}

    def allocate_sequence(self, seq_id: int, num_tokens: int):
        num_needed_blocks = (num_tokens + self.block_size - 1) // self.block_size
        if len(self.free_blocks) < num_needed_blocks:
            raise RuntimeError("Out of GPU KV-Cache Blocks!")
            
        allocated = [self.free_blocks.pop(0) for _ in range(num_needed_blocks)]
        self.block_tables[seq_id] = allocated

    def append_token(self, seq_id: int, current_seq_len: int):
        """Allocates a new physical block if current block is full."""
        if current_seq_len % self.block_size == 0:
            if not self.free_blocks:
                raise RuntimeError("Out of GPU KV-Cache Blocks during token generation!")
            self.block_tables[seq_id].append(self.free_blocks.pop(0))

    def get_block_table_tensor(self, seq_ids: List[int], max_blocks_per_seq: int) -> torch.Tensor:
        """Pads and returns block table as a 2D CUDA tensor [batch_size, max_blocks]."""
        batch_size = len(seq_ids)
        tensor = torch.zeros((batch_size, max_blocks_per_seq), dtype=torch.int32, device="cuda")
        
        for i, seq_id in enumerate(seq_ids):
            blocks = self.block_tables[seq_id]
            tensor[i, :len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
            
        return tensor
