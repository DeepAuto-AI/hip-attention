import torch

def flat_csr_to_dense(csr: torch.Tensor, T_SRC, H):
    # flatten CSR allows different number of element per batch
    assert csr.is_sparse_csr
    N, T_DST, H_T = csr.shape
    crow_indices = csr.crow_indices()
    col_indices = csr.col_indices()
    values = csr.values()
    denses = []
    for i in range(N):
        crow = crow_indices[i:i+1]
        col = col_indices[i:i+1]
        n_trim = 0
        for j in range(col.shape[-1]):
            if col[0, -(j+1)] == -1:
                n_trim += 1
            else:
                break
        # print(n_trim)
        v = values[i:i+1]
        if n_trim > 0:
            col = col[:, :-n_trim]
            v = v[:, :-n_trim]
        assert col.shape == v.shape
        # print(crow.shape, col.shape, v.shape, csr.shape)
        if col.shape[-1] == 0:
            t_d = torch.zeros((1, H, T_DST, T_SRC), dtype=v.dtype, device=v.device)
        else:
            mini_csr = torch.sparse_csr_tensor(
                crow, col, v, (1, T_DST, H_T)
            )
            t_d = mini_csr.to_dense()
            t_d = t_d.view(1, T_DST, H, T_SRC).transpose(1, 2).contiguous()
        denses.append(t_d)
    return torch.cat(denses, dim=0)