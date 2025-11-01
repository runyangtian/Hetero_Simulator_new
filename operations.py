# Define all ops

import numpy as np
from typing import List, Dict
from hardware_models import TensorShape

class Op:
    def __init__(self, name: str):
        self.name = name

    def required_tensors(self) -> List[str]:
        return []

class UnaryOp(Op):  # C=f(A), e.g. C = SoftMax(A)
    def __init__(self, kind: str, A: str, C: str):
        super().__init__(f"{kind}:{A}->{C}")
        self.kind, self.A, self.C = kind.upper(), A, C

    def flops(self, shapes: Dict[str, TensorShape]) -> int:
        return int(np.prod(shapes[self.A].dims))    # return the number of flops this Op needs to execute.

    def required_tensors(self) -> List[str]:
        return [self.A]

class BinaryOp(Op): # C=f(A,B), e.g. C = A .* B (element-wise)
    def __init__(self, kind: str, A: str, B: str, C: str):
        super().__init__(f"{kind}:{A},{B}->{C}")
        self.kind, self.A, self.B, self.C = kind.upper(), A, B, C

    def flops(self, shapes: Dict[str, TensorShape]) -> int:
        dimsA = shapes[self.A].dims
        dimsB = shapes[self.B].dims     
        assert dimsA[-2:] == dimsB[-2:], f"BinaryOp input shapes not compatible: {dimsA} vs {dimsB}"
        return int(np.prod(shapes[self.A].dims))

    def required_tensors(self) -> List[str]:
        return [self.A, self.B]

class ParallelOp(Op):
    def __init__(self, branches: List[Op]):
        super().__init__(f"Parallel({','.join(b.name for b in branches)})")
        self.branches = branches

    def flops(self, shapes: Dict[str, TensorShape]) -> int:
        return sum(b.flops(shapes) for b in self.branches) # recursively calculate

    def required_tensors(self) -> List[str]:
        reqs = []
        for b in self.branches:
            reqs.extend(b.required_tensors())
        return reqs
    
class UCIeOp:
    def __init__(self, size_bits):
        self.size_bits = size_bits

    def __repr__(self):
        return f"UCIeOp(bits={self.size_bits})"
    
# becides those above, you can define your own ops, such as
class MatMul(Op):   
    def __init__(self, A: str, B: str, C: str, transpose_B: bool = False):
        super().__init__(f"MatMul:{A}x{B}->{C}")
        self.A, self.B, self.C = A, B, C
        self.transpose_B = transpose_B

    def flops(self, shapes: Dict[str, TensorShape]) -> int:
        M, K = shapes[self.A].dims
        if self.transpose_B:
            N, K2 = shapes[self.B].dims
        else:
            K2, N = shapes[self.B].dims
        assert K == K2
        return M * N * K

    def required_tensors(self) -> List[str]:
        return [self.A, self.B]

# example of inheriting UnaryOp or BinaryOp
class SoftmaxOp(UnaryOp):
    def __init__(self, input_tensor: str, axis: int, output_tensor: str):
        super().__init__('SOFTMAX', input_tensor, output_tensor)
        self.axis = axis

class MulOp(BinaryOp):
    def __init__(self, A: str, B: str, C: str):
        super().__init__('MUL', A, B, C)