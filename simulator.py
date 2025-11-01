import math
import numpy as np
from typing import List, Dict, Any, Tuple
from model import Model
from hardware_models import MemoryDevice, ComputeUnit, Stats
from operations import (
    MatMul, UCIeOp, ParallelOp
)


class Simulator:
    def __init__(self, model: Model, schedule: List[Dict[str, Any]], rram: MemoryDevice, dram: MemoryDevice, dram_cu: ComputeUnit, rram_cu: ComputeUnit):
        self.model = model
        self.schedule = schedule
        self.rram, self.dram = rram, dram
        self.dram_cu = dram_cu
        self.rram_cu = rram_cu
        self.stats = Stats()
        self.ucie_bandwidth = 2048      # 32 Gb/s × 64 = 2,048 Gb/s；1 GHz → 2,048 Gb/s ÷ 1e9 = 2,048 bit/cycle
        # self.ucie_bandwidth = float("inf")
        self.ucie_energy_per_bit = 0.5  # pJ/bit
        # self.ucie_energy_per_bit = 0
        self.layer_latency_max_cycles = 3 # 0.01ns/layer, 256 layer in total
    
    # accumulate read/write energy and cycle
    def _acc_rw(self, dev: MemoryDevice, cycles: int, energy_nj: float, is_read: bool):
        if dev is self.dram:
            if is_read:
                # self.stats.cycles_read_dram += cycles
                self.stats.energy_read_dram_nj += energy_nj
            else:
                # self.stats.cycles_write_dram += cycles
                self.stats.energy_write_dram_nj += energy_nj
        elif dev is self.rram:
            if is_read:
                # self.stats.cycles_read_rram += cycles
                self.stats.energy_read_rram_nj += energy_nj
            else:
                # self.stats.cycles_write_rram += cycles
                self.stats.energy_write_rram_nj += energy_nj

    # accumulate computation energy and cycle
    def _acc_comp(self, cu: ComputeUnit, cycles: int, energy_nj: float):
        if cu is self.dram_cu:
            self.stats.cycles_comp_dram += cycles
            self.stats.energy_comp_dram_nj += energy_nj
        elif cu is self.rram_cu:
            self.stats.cycles_comp_rram += cycles
            self.stats.energy_comp_rram_nj += energy_nj

    # calculate mem read cost
    def _mem_read_cost(self, dev: MemoryDevice, size_bits: int, src_layer: int = -1):
        bw_cycles = math.ceil(size_bits / dev.read_bw_bits_per_cycle) if dev.read_bw_bits_per_cycle > 0 else 0
        cycles = dev.read_latency_cycles + bw_cycles
        energy = size_bits * dev.read_energy_per_bit

        hops = dev.tsv_hops(src_layer)
        tsv_cycles = dev.tsv_cycles_for(size_bits, hops)
        if hops==0:
            return 0, 0.0
        else:
            return cycles + tsv_cycles, energy

    # calculate mem write cost
    def _mem_write_cost(self, dev: MemoryDevice, size_bits: int, src_layer: int = -1):
        bw_cycles = math.ceil(size_bits / dev.write_bw_bits_per_cycle) if dev.write_bw_bits_per_cycle > 0 else 0
        cycles = dev.write_latency_cycles + bw_cycles
        energy = size_bits * dev.write_energy_per_bit

        hops = dev.tsv_hops(src_layer)
        tsv_cycles = dev.tsv_cycles_for(size_bits, hops)
        if hops==0:
            return 0, 0.0
        else:
            return cycles + tsv_cycles, energy

    # calculate computation cost
    def _compute_cost_engine(self, amount_ops: int, engine: str, cu: ComputeUnit):
        if amount_ops <= 0:
            return 0, 0.0

        if engine == 'sfe':
            cycles = (amount_ops + cu.sfe_ops_per_cycle - 1) // cu.sfe_ops_per_cycle
            energy = amount_ops * cu.sfe_energy_per_op_nj
        else:  # 'mac'
            cycles = (amount_ops + cu.macs_per_cycle - 1) // cu.macs_per_cycle
            energy = amount_ops * cu.energy_per_mac_nj

        return cycles, energy

    # get mem_device, layer, size_bits
    def _dev_and_layer(self, tname: str):
        t = self.model.tensors[tname]
        mem = self.rram if t.device == 'rram' else self.dram
        return mem, t.layer, t.bits_per_element

    def _calculate_item_cost(self, item: Dict[str, Any]) -> Tuple[int, float, int, int, int]:
        ttype = item['type']
        op = item.get('op')
        cycles, energy, macs, bits_read, bits_written = 0, 0, 0, 0, 0

        # define your own ops

        if ttype == 'matmul_tile':
            m, n, k = item['msize'], item['nsize'], item['ksize']

            memA, layerA, bpeA = self._dev_and_layer(op.A)
            memB, layerB, bpeB = self._dev_and_layer(op.B)
            memC, layerC, bpeC = self._dev_and_layer(op.C)

            # print(bpeA, bpeB, bpeC)
            A_tile_bits = m * k * bpeA
            B_tile_bits = k * n * bpeB
            C_tile_bits = m * n * bpeC

            # --- read A/B---
            cA, eA = self._mem_read_cost(memA, A_tile_bits, src_layer=layerA)
            cB, eB = self._mem_read_cost(memB, B_tile_bits, src_layer=layerB)
            c_in = max(cA, cB)

            # --- select CU：based on C's device---
            use_rram = (memC is self.rram)
            cu_sel = self.rram_cu if use_rram else self.dram_cu


            macs = m * n * k
            cc, ec = self._compute_cost_engine(macs, 'mac', cu_sel)
            cW, eW = self._mem_write_cost(memC, C_tile_bits, src_layer=layerC)

            self._acc_rw(memA, c_in, eA, is_read=True)
            self._acc_rw(memB, 0, eB, is_read=True)
            self._acc_comp(cu_sel, cc, ec)
            self._acc_rw(memC, cW, eW, is_read=False)
            
            cycles = max(c_in, cc, cW)
            energy = eA + eB + ec + eW

            bits_read = A_tile_bits + B_tile_bits
            bits_written = C_tile_bits

        elif ttype == 'ucieop':
            op = item['op']
            cycles = math.ceil(op.size_bits / max(1, self.ucie_bandwidth))  # 用 ceil 避免被截断为 0
            energy = op.size_bits * self.ucie_energy_per_bit * 1e-3         # pJ → nJ
            
        else:
            raise ValueError(f"Unknown ttype '{ttype}' in schedule item: {item}")

        return cycles, energy, macs, bits_read, bits_written

    def _simulate_parallel(self, parallel_item: Dict[str, Any]):
        branch_stats = []  # (b_cycles, b_energy, b_macs, b_br, b_bw, b_cycles_by_type)

        for branch_schedule in parallel_item['branches']:
            b_cycles = b_energy = b_macs = b_br = b_bw = 0
            b_cycles_by_type: Dict[str, int] = {}

            for item_in_branch in branch_schedule:
                # if parallel op is still in the branch, do it again
                if item_in_branch.get('type') == 'parallel':
                    c2, e2, m2, br2, bw2, cbt2 = self._simulate_parallel(item_in_branch)
                    b_cycles += c2
                    b_energy += e2
                    b_macs   += m2
                    b_br     += br2
                    b_bw     += bw2
                    for t, cpart in cbt2.items():
                        b_cycles_by_type[t] = b_cycles_by_type.get(t, 0) + cpart
                    continue

                c, e, m, br, bw = self._calculate_item_cost(item_in_branch)
                b_cycles += c
                b_energy += e
                b_macs   += m
                b_br     += br
                b_bw     += bw

                # energy and MACs added into each ops' catagory
                t = item_in_branch['type']
                self.stats.breakdown[t]      = self.stats.breakdown.get(t, 0) + e
                self.stats.macs_breakdown[t] = self.stats.macs_breakdown.get(t, 0) + m

                # calculate the latency for this brance
                b_cycles_by_type[t] = b_cycles_by_type.get(t, 0) + c

            branch_stats.append((b_cycles, b_energy, b_macs, b_br, b_bw, b_cycles_by_type))

        if not branch_stats:
            return 0, 0, 0, 0, 0, {}

        # only accumlate the longest latency among each branches
        winner = max(branch_stats, key=lambda x: x[0])

        # for parallel op, only use the longest latency, but accumlate energy, macs, ...
        cycles       = winner[0]
        energy       = sum(s[1] for s in branch_stats)
        macs         = sum(s[2] for s in branch_stats)
        bits_read    = sum(s[3] for s in branch_stats)
        bits_written = sum(s[4] for s in branch_stats)

        # update
        self.stats.cycles       += cycles
        self.stats.energy_nj    += energy
        self.stats.macs         += macs
        self.stats.bits_read    += bits_read
        self.stats.bits_written += bits_written

        winner_cycles_by_type = winner[5]
        for t, cpart in winner_cycles_by_type.items():
            self.stats.cycles_breakdown[t] = self.stats.cycles_breakdown.get(t, 0) + cpart

        return cycles, energy, macs, bits_read, bits_written, winner_cycles_by_type

    def run(self):
        for item in self.schedule:
            ttype = item['type']
            if ttype == 'parallel':
                _ = self._simulate_parallel(item)
            else:
                cycles, energy, macs, bits_read, bits_written = self._calculate_item_cost(item)

                self.stats.cycles       += cycles
                self.stats.energy_nj    += energy
                self.stats.macs         += macs
                self.stats.bits_read    += bits_read
                self.stats.bits_written += bits_written

                self.stats.breakdown[ttype]        = self.stats.breakdown.get(ttype, 0) + energy
                self.stats.macs_breakdown[ttype]   = self.stats.macs_breakdown.get(ttype, 0) + macs
                self.stats.cycles_breakdown[ttype] = self.stats.cycles_breakdown.get(ttype, 0) + cycles

        return self.stats
