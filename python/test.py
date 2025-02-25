import torch

from sglang.srt.layers.attention.triton_ops.extend_attention import extend_attention_fwd

q_extend = torch.load("./tensors/q_extend.pt").cuda(0)
k_extend = torch.load("./tensors/k_extend.pt").cuda(0)
v_extend = torch.load("./tensors/v_extend.pt").cuda(0)
o_extend = torch.load("./tensors/o_extend.pt").cuda(0)
k_buffer = torch.load("./tensors/k_buffer.pt").cuda(0)
v_buffer = torch.load("./tensors/v_buffer.pt").cuda(0)
qo_indptr = torch.load("./tensors/qo_indptr.pt").cuda(0)
kv_indptr = torch.load("./tensors/kv_indptr.pt").cuda(0)
kv_indices = torch.load("./tensors/kv_indices.pt").cuda(0)
custom_mask = None
mask_indptr = None
max_len_extend = torch.load("./tensors/max_len_extend.pt")
for i in range(10):
    extend_attention_fwd(
        q_extend,
        k_extend,
        v_extend,
        o_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        custom_mask,
        mask_indptr,
        max_len_extend,
        0.1352337788608801,
    )

print(o_extend)
