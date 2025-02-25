import torch


class AngularLSH(torch.nn.Module):

    def __init__(self, num_projs, dim, rng=None):
        super().__init__()
        self.num_projs = num_projs

        if num_projs > 0:
            self.register_buffer(
                "proj_dir",
                torch.randn(dim + (num_projs,), generator=rng),
                persistent=False,
            )
            self.register_buffer(
                "perm",
                self._unit_hamming_distance_array(self.num_projs),
                persistent=False,
            )
            self.register_buffer(
                "enc_vec",
                2 ** torch.arange(self.num_projs).view(1, 1, 1, -1),
                persistent=False,
            )

    def _unit_hamming_distance_array(self, size_n):
        if size_n == 1:
            return torch.tensor([0, 1])
        a = self._unit_hamming_distance_array(size_n - 1)
        return torch.concat([a, torch.flip(a, dims=[0]) + 2 ** (size_n - 1)], 0)

    def hash(self, mat):
        if self.num_projs < 0:
            return torch.zeros(mat.shape[:-1], device=mat.device, dtype=torch.int32)
        mask = torch.einsum("...nd,...dr -> ...nr", mat, self.proj_dir)
        mask = mask > 0
        bin_ids = (mask * self.enc_vec).sum(-1)
        return self.perm[bin_ids]

    def __repr__(self):
        return f"AngularLSH(num_proj={self.num_projs}, proj_dir.shape={self.proj_dir.shape})"
