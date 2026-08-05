from typing import List, Tuple

import torch

from engine.models.llama import Llama3BEngineModel


class CUDAGraphRunner:
    def __init__(self, model: Llama3BEngineModel, max_batch_size: int = 16):
        self.model = model
        self.max_batch_size = max_batch_size
        self.device = "cuda"

        # Pre-allocate static graph input/output tensors using max_batch_size
        self.static_input_ids = torch.zeros(
            (max_batch_size, 1), dtype=torch.int64, device=self.device
        )
        self.static_positions = torch.zeros(
            (max_batch_size, 1), dtype=torch.int32, device=self.device
        )
        self.static_context_lens = torch.zeros(
            (max_batch_size,), dtype=torch.int32, device=self.device
        )
        self.static_slot_mapping = torch.zeros(
            (max_batch_size,), dtype=torch.int32, device=self.device
        )
        self.static_block_tables = torch.zeros(
            (max_batch_size, 1024), dtype=torch.int32, device=self.device
        )  # Assume max 1024 blocks
        self.static_logits = torch.zeros(
            (max_batch_size, self.model.config.vocab_size),
            dtype=torch.float16,
            device=self.device,
        )

        self.cuda_graph: torch.cuda.CUDAGraph = None
        self.is_captured: bool = False

    def _prepare_inputs(
        self, seq_ids: List[int], context_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        batch_size = len(seq_ids)
        max_blocks = (
            context_lens.max().item() + self.model.cache_manager.block_size - 1
        ) // self.model.cache_manager.block_size

        slot_mapping = torch.zeros((batch_size,), dtype=torch.int32, device=self.device)
        for i, seq_id in enumerate(seq_ids):
            curr_pos = context_lens[i].item() - 1
            b_idx = self.model.cache_manager.block_tables[seq_id][
                curr_pos // self.model.cache_manager.block_size
            ]
            b_offset = curr_pos % self.model.cache_manager.block_size
            slot_mapping[i] = b_idx * self.model.cache_manager.block_size + b_offset

        block_tables = self.model.cache_manager.get_block_table_tensor(
            seq_ids, max_blocks
        )
        return slot_mapping, block_tables, max_blocks

    def capture_graph(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        seq_ids: List[int],
        context_lens: torch.Tensor,
    ):
        batch_size = input_ids.size(0)
        print(f"⚡ Capturing CUDA Graph for batch size = {batch_size}...")

        slot_mapping, block_tables, max_blocks = self._prepare_inputs(
            seq_ids, context_lens
        )
        self.max_blocks_per_seq = max_blocks

        # 1. Allocate exact-sized static tensors for captured batch size (no slicing views!)
        self.static_input_ids = torch.zeros(
            (batch_size, 1), dtype=torch.int64, device=self.device
        )
        self.static_positions = torch.zeros(
            (batch_size, 1), dtype=torch.int32, device=self.device
        )
        self.static_context_lens = torch.zeros(
            (batch_size,), dtype=torch.int32, device=self.device
        )
        self.static_slot_mapping = torch.zeros(
            (batch_size,), dtype=torch.int32, device=self.device
        )
        self.static_block_tables = torch.zeros(
            (batch_size, max_blocks), dtype=torch.int32, device=self.device
        )
        self.static_logits = torch.zeros(
            (batch_size, self.model.config.vocab_size),
            dtype=torch.float16,
            device=self.device,
        )

        # Use pre-allocated static tensors but update their contents via copy_
        # Ensure input_ids/positions etc match the pre-allocated shape.
        batch_size = input_ids.size(0)
        self.static_input_ids[:batch_size].copy_(input_ids)
        self.static_positions[:batch_size].copy_(positions)
        self.static_context_lens[:batch_size].copy_(context_lens)
        self.static_slot_mapping[:batch_size].copy_(slot_mapping)
        self.static_block_tables[:batch_size, :max_blocks].copy_(block_tables)

        kv_backup = self.model.cache_manager.kv_cache.clone()

        # 1. Warmup Stream
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())

        # NOTE: Ensure the model operations are fully synchronous if necessary
        # or that they don't depend on non-static data.
        with torch.cuda.stream(s):
            for _ in range(5):  # Increase warmup iterations
                _ = self.model.decode_step(
                    self.static_input_ids[:batch_size],
                    self.static_positions[:batch_size],
                    self.static_context_lens[:batch_size],
                    self.static_slot_mapping[:batch_size],
                    self.static_block_tables[:batch_size, :max_blocks],
                    self.max_blocks_per_seq,
                )
        s.synchronize()

        self.model.cache_manager.kv_cache.copy_(kv_backup)
        torch.cuda.synchronize()

        # 3. Capture Graph
        self.cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(s):
            with torch.cuda.graph(self.cuda_graph, stream=s):
                out = self.model.decode_step(
                    self.static_input_ids[:batch_size],
                    self.static_positions[:batch_size],
                    self.static_context_lens[:batch_size],
                    self.static_slot_mapping[:batch_size],
                    self.static_block_tables[:batch_size, :max_blocks],
                    self.max_blocks_per_seq,
                )
                self.static_logits[:batch_size].copy_(out)

        s.synchronize()

        self.model.cache_manager.kv_cache.copy_(kv_backup)
        torch.cuda.synchronize()

        self.is_captured = True
        print("✅ CUDA Graph Capture Complete!")

    def execute(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        seq_ids: List[int],
        context_lens: torch.Tensor,
    ) -> torch.Tensor:
        if self.is_captured:
            batch_size = input_ids.size(0)
            slot_mapping, block_tables, max_blocks = self._prepare_inputs(
                seq_ids, context_lens
            )

            # Copy directly into pre-allocated static buffers using views for batch_size
            self.static_input_ids[:batch_size].copy_(input_ids)
            self.static_positions[:batch_size].copy_(positions)
            self.static_context_lens[:batch_size].copy_(context_lens)
            self.static_slot_mapping[:batch_size].copy_(slot_mapping)
            self.static_block_tables[:batch_size, :max_blocks].copy_(block_tables)

            self.cuda_graph.replay()
            return self.static_logits[:batch_size]
        else:
            slot_mapping, block_tables, max_blocks = self._prepare_inputs(
                seq_ids, context_lens
            )
            return self.model.decode_step(
                input_ids,
                positions,
                context_lens,
                slot_mapping,
                block_tables,
                max_blocks,
            )
