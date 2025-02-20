import numpy as np
from intel_npu_acceleration_library.backend import MatMul

inC, outC, batch = 4096, 4096, 32

# Create both inputs
X1 = np.random.uniform(-1, 1, (batch, inC)).astype(np.float16)
X2 = np.random.uniform(-1, 1, (outC, inC)).astype(np.float16)

mm = MatMul(inC, outC, batch, profile=False)

result = mm.run(X1, X2)

print(result)
