#!/usr/bin/bash

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

python tests/v1_2/test_latency_e2e.py --method minf
python tests/v1_2/test_latency_e2e.py --method delta2-estk
python tests/v1_2/test_latency_e2e.py --method delta2-exact
python tests/v1_2/test_latency_e2e.py --method hip
python tests/v1_2/test_latency_e2e.py --method fa3
python tests/v1_2/test_latency_e2e.py --method fa2
