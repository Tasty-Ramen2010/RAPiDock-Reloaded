##########################################################################
# File Name: metal_tp.py
# Fused Metal (Apple GPU) kernel for e3nn Clebsch-Gordan tensor products
#########################################################################
"""Fused Metal compute kernel for e3nn ``FullyConnectedTensorProduct``.

Why this exists
---------------
Stock e3nn lowers a *single* tensor product into roughly 370 individual ``aten``
dispatches -- one cluster of ``einsum`` / ``bmm`` / ``permute`` / ``reshape`` per
instruction path.  On CUDA that is largely hidden by cheap kernel launches.  On
Apple's MPS backend it is not: the dispatch overhead plus the round-trips
through materialised intermediates leave the layer running far below the GPU's
memory bandwidth.  Measured on an M3 (10-core GPU), the four conv layers of
``CGTensorProductEquivariantModel`` cost ~67 ms per pass at E=20000 edges, while
the unavoidable memory traffic implies a floor closer to ~6 ms.

This module JIT-generates one Metal kernel, specialised to a single tensor
product configuration, that evaluates the whole product in one dispatch and
reads the per-edge weight tensor -- by far the largest operand -- exactly once.

The maths mirrors e3nn's own codegen exactly (see ``e3nn/o3/_tensor_product``)::

    out[z, w, k] = path_weight * sum_{u,v,i,j}
                       W[z, u, v, w] * w3j[i, j, k] * x1[z, u, i] * x2[z, v, j]

``path_weight`` is the ``sqrt(alpha)`` normalisation e3nn precomputes and stores
on each ``Instruction``; ``w3j`` is ``o3.wigner_3j(l1, l2, l_out)`` -- the same
function e3nn itself calls, so there is no re-derivation of conventions here.
e3nn's ``specialized_code`` shortcuts for the ``l == 0`` cases are algebraically
identical to the general ``w3j`` contraction, so only the general form is
emitted.

Thread mapping
--------------
One thread per ``(edge, output multiplicity slot)``, where a slot is a
``(output irrep, multiplicity index)`` pair.  Each thread therefore produces all
``2*l_out+1`` components of one output multiplicity and consumes each weight
element exactly once.  Mapping one thread per output *element* instead would be
simpler but would re-read the weight tensor ``2*l_out+1`` times.

Safety
------
:class:`FusedTensorProduct` is a drop-in accelerator that owns a reference to
the original e3nn module and falls back to it whenever anything is off-nominal:
non-MPS tensors, non-float32 dtypes, unexpected shapes, an unsupported
configuration, or -- importantly -- whenever autograd is active, since this
kernel implements no backward pass.  It is deliberately *not* an ``nn.Module``
so that wiring it into a model leaves ``state_dict`` keys untouched and
pretrained checkpoints keep loading.

Set ``RAPIDOCK_DISABLE_METAL_TP=1`` to disable it entirely.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import torch

try:
    from e3nn import o3
except ImportError:  # pragma: no cover - e3nn is a hard dep of the model anyway
    o3 = None


__all__ = ["FusedTensorProduct", "maybe_fuse", "metal_tp_available"]


# Compiled shader libraries are cached on the generated source, so the many
# structurally identical conv layers share one compilation.
_LIB_CACHE: dict = {}


class _Unsupported(Exception):
    """Raised during planning when a tensor product cannot be fused."""


def metal_tp_available() -> bool:
    """True if this build of torch can compile Metal shaders on this machine."""
    if os.environ.get("RAPIDOCK_DISABLE_METAL_TP", "") == "1":
        return False
    if o3 is None:
        return False
    if not hasattr(torch, "mps") or not hasattr(torch.mps, "compile_shader"):
        return False
    try:
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def _lit(x: float) -> str:
    """Format a python float as an exactly-round-tripping MSL float literal."""
    return f"{float(x):.9e}f"


class _Instruction:
    """One fusable ``uvw`` path, with everything resolved to flat offsets."""

    __slots__ = ("i_out", "mul1", "mul2", "mulo", "d1", "d2", "do",
                 "x1_off", "x2_off", "w_off", "pw", "nz")

    def __init__(self, i_out, mul1, mul2, mulo, d1, d2, do,
                 x1_off, x2_off, w_off, pw, nz):
        self.i_out = i_out
        self.mul1, self.mul2, self.mulo = mul1, mul2, mulo
        self.d1, self.d2, self.do = d1, d2, do
        self.x1_off, self.x2_off, self.w_off = x1_off, x2_off, w_off
        self.pw = pw
        self.nz = nz  # list of (i, j, k, value) Wigner-3j nonzeros


def _irrep_offsets(irreps) -> List[int]:
    offs, acc = [], 0
    for mul_ir in irreps:
        offs.append(acc)
        acc += mul_ir.dim
    return offs


def _plan(tp) -> Tuple[List[_Instruction], List[int], List[int], int, int, int, int]:
    """Resolve an e3nn TensorProduct into flat, kernel-ready metadata."""
    if getattr(tp, "shared_weights", False):
        raise _Unsupported("shared_weights=True is not supported")
    if getattr(tp, "in1_var", None) is None:
        pass  # older/newer e3nn; not needed, normalisation lives in path_weight

    irreps_in1, irreps_in2, irreps_out = tp.irreps_in1, tp.irreps_in2, tp.irreps_out
    off1 = _irrep_offsets(irreps_in1)
    off2 = _irrep_offsets(irreps_in2)
    offo = _irrep_offsets(irreps_out)

    instructions: List[_Instruction] = []
    flat_w = 0
    for ins in tp.instructions:
        mul_ir_1 = irreps_in1[ins.i_in1]
        mul_ir_2 = irreps_in2[ins.i_in2]
        mul_ir_o = irreps_out[ins.i_out]

        if ins.connection_mode != "uvw":
            raise _Unsupported(f"connection_mode {ins.connection_mode!r} not supported")
        if not ins.has_weight:
            raise _Unsupported("weightless instructions are not supported")

        n_w = int(mul_ir_1.mul) * int(mul_ir_2.mul) * int(mul_ir_o.mul)
        w_off = flat_w
        flat_w += n_w

        # e3nn skips empty paths entirely (and so must the weight accounting
        # above, which mirrors its `flat_weight_index` bookkeeping).
        if mul_ir_1.dim == 0 or mul_ir_2.dim == 0 or mul_ir_o.dim == 0:
            continue

        l1, l2, lo = mul_ir_1.ir.l, mul_ir_2.ir.l, mul_ir_o.ir.l
        w3j = o3.wigner_3j(l1, l2, lo).to(torch.float64)
        nz = []
        for i in range(w3j.shape[0]):
            for j in range(w3j.shape[1]):
                for k in range(w3j.shape[2]):
                    v = float(w3j[i, j, k])
                    if v != 0.0:
                        nz.append((i, j, k, v))
        if not nz:
            continue

        instructions.append(_Instruction(
            i_out=ins.i_out,
            mul1=int(mul_ir_1.mul), mul2=int(mul_ir_2.mul), mulo=int(mul_ir_o.mul),
            d1=2 * l1 + 1, d2=2 * l2 + 1, do=2 * lo + 1,
            x1_off=off1[ins.i_in1], x2_off=off2[ins.i_in2], w_off=w_off,
            pw=float(ins.path_weight), nz=nz,
        ))

    if flat_w != int(tp.weight_numel):
        raise _Unsupported(
            f"weight layout mismatch: derived {flat_w}, e3nn reports {tp.weight_numel}"
        )
    if not instructions:
        raise _Unsupported("no fusable instructions")

    # Output slots: one per (output irrep, multiplicity index), in output order.
    slot_io: List[int] = []
    slot_w: List[int] = []
    for i_out, mul_ir in enumerate(irreps_out):
        for w in range(int(mul_ir.mul)):
            slot_io.append(i_out)
            slot_w.append(w)
    if not slot_io:
        raise _Unsupported("empty output irreps")
    if len(slot_io) > 255 and max(slot_io) > 255:
        raise _Unsupported("too many output irreps for uchar slot table")
    if max(slot_w) > 65535:
        raise _Unsupported("multiplicity exceeds ushort slot table")

    return (instructions, slot_io, slot_w,
            irreps_in1.dim, irreps_in2.dim, irreps_out.dim, int(tp.weight_numel))


def _emit_case(i_out: int, out_off: int, do: int,
               ins_list: List[_Instruction]) -> str:
    """Emit the fully-unrolled body for one output irrep."""
    body: List[str] = []
    body.append(f"    case {i_out}u: {{")
    for k in range(do):
        body.append(f"        float a{k} = 0.0f;")

    for n, ins in enumerate(ins_list):
        body.append(f"        /* path {n}: mul1={ins.mul1} d1={ins.d1} x "
                    f"mul2={ins.mul2} d2={ins.d2} -> mulo={ins.mulo} do={ins.do} */")
        body.append("        {")
        for v in range(ins.mul2):
            # X2 is loop-invariant in u, so fold it into the Wigner coefficient
            # once per path instead of once per multiplicity.
            for t, (i, j, k, val) in enumerate(ins.nz):
                body.append(
                    f"            const float c{v}_{t} = {_lit(val)} * "
                    f"X2[{ins.x2_off + v * ins.d2 + j}u];"
                )
        body.append(f"            for (uint u = 0u; u < {ins.mul1}u; ++u) {{")
        body.append(f"                const uint xb = {ins.x1_off}u + u * {ins.d1}u;")
        for v in range(ins.mul2):
            body.append(
                f"                const float g{v} = "
                f"Wp[{ins.w_off + v * ins.mulo}u + w + u * {ins.mul2 * ins.mulo}u]"
                f" * {_lit(ins.pw)};"
            )
        for v in range(ins.mul2):
            by_k: dict = {}
            for t, (i, j, k, val) in enumerate(ins.nz):
                by_k.setdefault(k, []).append(f"c{v}_{t} * X1[xb + {i}u]")
            for k in sorted(by_k):
                terms = " + ".join(by_k[k])
                body.append(f"                a{k} += g{v} * ({terms});")
        body.append("            }")
        body.append("        }")

    body.append(f"        device float* dst = O + {out_off}u + w * {do}u;")
    for k in range(do):
        body.append(f"        dst[{k}] = a{k};")
    body.append("        break;")
    body.append("    }")
    return "\n".join(body)


def _emit_zero_case(i_out: int, out_off: int, do: int) -> str:
    lines = [f"    case {i_out}u: {{",
             f"        device float* dst = O + {out_off}u + w * {do}u;"]
    for k in range(do):
        lines.append(f"        dst[{k}] = 0.0f;")
    lines.append("        break;")
    lines.append("    }")
    return "\n".join(lines)


def _generate_msl(tp, instructions, slot_io, slot_w,
                  in1_dim, in2_dim, out_dim, wnumel) -> str:
    irreps_out = tp.irreps_out
    offo = _irrep_offsets(irreps_out)

    by_out: dict = {}
    for ins in instructions:
        by_out.setdefault(ins.i_out, []).append(ins)

    cases = []
    for i_out, mul_ir in enumerate(irreps_out):
        if int(mul_ir.mul) == 0:
            continue
        do = 2 * mul_ir.ir.l + 1
        if i_out in by_out:
            cases.append(_emit_case(i_out, offo[i_out], do, by_out[i_out]))
        else:
            # A declared output irrep that no instruction feeds: e3nn emits
            # zeros for it, so we must too.
            cases.append(_emit_zero_case(i_out, offo[i_out], do))

    nslot = len(slot_io)
    io_tbl = ", ".join(str(x) for x in slot_io)
    w_tbl = ", ".join(str(x) for x in slot_w)

    return f"""#include <metal_stdlib>
using namespace metal;

/* Generated by RAPiDock utils/metal_tp.py -- do not edit by hand.
   in1={tp.irreps_in1}  in2={tp.irreps_in2}  out={tp.irreps_out}
   weight_numel={wnumel}  slots={nslot} */

constant uchar  SLOT_IO[{nslot}] = {{ {io_tbl} }};
constant ushort SLOT_W [{nslot}] = {{ {w_tbl} }};

kernel void fused_tp(
        device float*       out [[buffer(0)]],
        device const float* x1  [[buffer(1)]],
        device const float* x2  [[buffer(2)]],
        device const float* wts [[buffer(3)]],
        constant uint&      E   [[buffer(4)]],
        uint gid [[thread_position_in_grid]])
{{
    const uint nslot = {nslot}u;
    const uint z = gid / nslot;
    if (z >= E) return;
    const uint slot = gid - z * nslot;
    const uint io = uint(SLOT_IO[slot]);
    const uint w  = uint(SLOT_W [slot]);

    device const float* X1 = x1  + z * {in1_dim}u;
    device const float* X2 = x2  + z * {in2_dim}u;
    device const float* Wp = wts + z * {wnumel}u;
    device float*       O  = out + z * {out_dim}u;

    switch (io) {{
{chr(10).join(cases)}
    }}
}}
"""


class FusedTensorProduct:
    """Drop-in Metal accelerator for an ``o3.FullyConnectedTensorProduct``.

    Deliberately not an ``nn.Module``: assigning one to a model attribute must
    not add ``state_dict`` keys, or pretrained checkpoints stop loading.
    """

    def __init__(self, tp):
        self.tp = tp
        (self._ins, self._slot_io, self._slot_w,
         self.in1_dim, self.in2_dim, self.out_dim, self.wnumel) = _plan(tp)
        self.nslot = len(self._slot_io)
        self.source = _generate_msl(
            tp, self._ins, self._slot_io, self._slot_w,
            self.in1_dim, self.in2_dim, self.out_dim, self.wnumel,
        )

    def _library(self):
        # Looked up from the module-level cache on every call rather than held
        # on the instance: a compiled shader handle is neither deep-copyable nor
        # picklable, and keeping one here would quietly break anything that
        # copies or serialises the enclosing model.
        cached = _LIB_CACHE.get(self.source)
        if cached is None:
            cached = torch.mps.compile_shader(self.source)
            _LIB_CACHE[self.source] = cached
        return cached

    def _eligible(self, x1, x2, w) -> bool:
        if torch.is_grad_enabled() and (
            x1.requires_grad or x2.requires_grad or w.requires_grad
        ):
            return False  # no backward implemented; never silently break training
        for t in (x1, x2, w):
            if t.device.type != "mps" or t.dtype != torch.float32 or t.dim() != 2:
                return False
        E = x1.shape[0]
        if x2.shape[0] != E or w.shape[0] != E:
            return False
        return (x1.shape[1] == self.in1_dim
                and x2.shape[1] == self.in2_dim
                and w.shape[1] == self.wnumel)

    def __call__(self, x1, x2, w):
        if not self._eligible(x1, x2, w):
            return self.tp(x1, x2, w)
        E = int(x1.shape[0])
        if E == 0:
            return torch.zeros((0, self.out_dim), device=x1.device, dtype=x1.dtype)
        x1 = x1.contiguous()
        x2 = x2.contiguous()
        w = w.contiguous()
        out = torch.empty((E, self.out_dim), device=x1.device, dtype=torch.float32)
        self._library().fused_tp(out, x1, x2, w, E, threads=E * self.nslot)
        return out


def maybe_fuse(tp) -> Optional[FusedTensorProduct]:
    """Build a fused kernel for ``tp``, or return None if it cannot be fused.

    Never raises: an unfusable or unsupported tensor product simply keeps using
    stock e3nn.
    """
    if not metal_tp_available():
        return None
    try:
        return FusedTensorProduct(tp)
    except Exception:
        return None
