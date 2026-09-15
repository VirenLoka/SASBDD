"""SA surrogate for DiffSBDD (stage 1).

Note on Apple silicon: PyG's TransformerConv aggregates through
aten::scatter_reduce, which the MPS backend does not implement.  Running on MPS
therefore requires PYTORCH_ENABLE_MPS_FALLBACK=1, and that variable is read when
torch initialises its backends -- setting it from here would be too late to have
any effect while still making the check in get_device() believe it was set.  So
it is deliberately NOT set here.  `device: auto` resolves to CUDA or CPU; ask for
MPS explicitly only if you also export the variable before launching.
"""
