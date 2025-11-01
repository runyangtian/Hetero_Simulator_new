# Parsing JSON files and building Model objects (add tensor and op)

import json
from typing import Dict, Any

from model import Model
from operations import (
    MatMul, UCIeOp, ParallelOp
)

class JSONModelLoader:
    def __init__(self, default_bits: int = 16):
        self.default_bits = default_bits

    def build(self, spec: Dict[str, Any]) -> Model:
        m = Model()
        # add tensor
        for t in spec.get("tensors", []):
            m.add_tensor(
                name=t["name"],
                shape=tuple(int(x) for x in t["shape"]),
                bits_per_element=int(t.get("bits", self.default_bits)),
                device=t.get("device"),
                layer=int(t.get("layer", -1)) 
            )

        for o in spec.get("ops", []):
            tpe = o.get("type", "").lower()

            # define your own ops
            if tpe == 'matmul':
                transpose_B = o.get('transpose_B', False)   # False by default
                m.add_op(MatMul(o['A'], o['B'], o['C'], transpose_B=transpose_B))

            elif tpe == 'ucieop':
                m.add_op(UCIeOp(int(o['size_bits'])))

            # parallel op
            elif tpe == 'parallelops':
                branches = []
                for branch in o['branches']:
                    branches.append(self.build({"ops": [branch], "tensors": []}).ops[0])
                m.add_op(ParallelOp(branches))

            else:
                raise ValueError(f"Unknown op type in JSON: {o}")
            
        return m
