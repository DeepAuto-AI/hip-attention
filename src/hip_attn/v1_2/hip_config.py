import json
import os
import warnings
from dataclasses import InitVar, dataclass, field
from typing import List, Optional, Union

from hip_attn.v1_2.attention_metadata import ScanStage

HIP_CONFIG_PRESET = os.getenv("HIP_CONFIG_PRESET", "default")

HIP_DEBUG_LANDMARK_BASED_SCAN_STAGE = (
    os.getenv("HIP_DEBUG_LANDMARK_BASED_SCAN_STAGE", "1") == "1"
)
HIP_DEBUG_DELTA_EXP = "exp" in os.getenv("HIP_DELTA_ATTENTION_ARGS", "")

if HIP_DEBUG_LANDMARK_BASED_SCAN_STAGE:
    if HIP_DEBUG_DELTA_EXP:
        _DEFAULT_STAGES = [
            ScanStage(
                stage_block_size_q=64,
                stage_block_stride_q=2,
                stage_chunk_size=64,
                stage_k=None,
                stage_stride=1,
            ),
            ScanStage(
                stage_block_size_q=64,
                stage_block_stride_q=2,
                stage_chunk_size=16,
                stage_k=32768,
                stage_stride=1,
            ),
            ScanStage(
                stage_block_size_q=64,
                stage_block_stride_q=1,
                stage_chunk_size=4,
                stage_k=8192,
                stage_stride=1,
            ),
        ]
    else:
        _DEFAULT_STAGES = [
            ScanStage(
                stage_block_size_q=64,
                stage_block_stride_q=4,
                stage_chunk_size=64,
                stage_k=None,
                stage_stride=1,
            ),
            ScanStage(
                stage_block_size_q=64,
                stage_block_stride_q=4,
                stage_chunk_size=16,
                stage_k=32768,
                stage_stride=1,
            ),
            ScanStage(
                stage_block_size_q=64,
                stage_block_stride_q=1,
                stage_chunk_size=4,
                stage_k=8192,
                stage_stride=1,
            ),
        ]
    _DEFAULT_STAGES_DECODE = [
        ScanStage(
            stage_block_size_q=64,
            stage_block_stride_q=4,
            stage_chunk_size=128,
            stage_k=None,
            stage_stride=1,
        ),
        ScanStage(
            stage_block_size_q=64,
            stage_block_stride_q=4,
            stage_chunk_size=32,
            stage_k=32768,
            stage_stride=1,
        ),
        ScanStage(
            stage_block_size_q=64,
            stage_block_stride_q=1,
            stage_chunk_size=8,
            stage_k=8192,
            stage_stride=1,
        ),
    ]
else:
    _DEFAULT_STAGES = [
        ScanStage(
            stage_block_size_q=64,
            stage_block_stride_q=4,
            stage_chunk_size=128,
            stage_k=None,
            stage_stride=1,
        ),
        ScanStage(
            stage_block_size_q=64,
            stage_block_stride_q=4,
            stage_chunk_size=32,
            stage_k=32768,
            stage_stride=1,
        ),
        ScanStage(
            stage_block_size_q=64,
            stage_block_stride_q=1,
            stage_chunk_size=8,
            stage_k=8192,
            stage_stride=1,
        ),
    ]


@dataclass
class HiPAttentionPerLayerConfig:
    second_stage_k: int = 2048
    sliding_window_size: int = 1024
    sliding_window_size_for_masking_step: Optional[List[int]] = None
    sink_token_size: int = 256
    landmark_stage_k: int = field(default_factory=lambda: [1, 1, 1])
    sa_extend_backend: str = "streaming"
    scan_extend_backend: Optional[str] = None
    stages: list[ScanStage] = field(default_factory=lambda: _DEFAULT_STAGES)

    parsed_json: InitVar[Optional[dict]] = None

    def __post_init__(self, parsed_json: Optional[dict]):
        super().__init__()
        if parsed_json is not None:
            if "second_stage_k" in parsed_json:
                self.second_stage_k = parsed_json["second_stage_k"]
                parsed_json.pop("second_stage_k")
            if "sliding_window_size" in parsed_json:
                self.sliding_window_size = parsed_json["sliding_window_size"]
                parsed_json.pop("sliding_window_size")
            if "sliding_window_size_for_masking_step" in parsed_json:
                self.sliding_window_size_for_masking_step = parsed_json[
                    "sliding_window_size_for_masking_step"
                ]
                parsed_json.pop("sliding_window_size_for_masking_step")
            if "sink_token_size" in parsed_json:
                self.sink_token_size = parsed_json["sink_token_size"]
                parsed_json.pop("sink_token_size")
            if "sa_extend_backend" in parsed_json:
                self.sa_extend_backend = parsed_json["sa_extend_backend"]
                parsed_json.pop("sa_extend_backend")
            if "scan_extend_backend" in parsed_json:
                self.scan_extend_backend = parsed_json["scan_extend_backend"]
                parsed_json.pop("scan_extend_backend")
            if "stages" in parsed_json:
                self.stages = [
                    (
                        ScanStage(**stage)
                        if len(stage.keys()) > 0
                        else ScanStage(64, 1, 32, 32768, 1)
                    )
                    for stage in parsed_json["stages"]
                ]
                parsed_json.pop("stages")
            if "landmark_stage_k" in parsed_json:
                self.landmark_stage_k = parsed_json["landmark_stage_k"]
                parsed_json.pop("landmark_stage_k")
            if parsed_json:
                raise ValueError(f"Unknown keys in json: {parsed_json.keys()}")


if HIP_CONFIG_PRESET == "default":
    _DEFAULT_LAEYRS = [
        HiPAttentionPerLayerConfig(
            # sliding_window_size = 777, # NOTE: debugging sw
            second_stage_k=4096,
            sa_extend_backend="streaming",
            scan_extend_backend="streaming",
            stages=_DEFAULT_STAGES,
        ),
        HiPAttentionPerLayerConfig(
            # sliding_window_size = 777, # NOTE: debugging sw
            sliding_window_size=1024,
            second_stage_k=2048,
            sa_extend_backend="streaming",
            scan_extend_backend="relative",
            stages=_DEFAULT_STAGES,
        ),
    ]
    if HIP_DEBUG_LANDMARK_BASED_SCAN_STAGE:
        _DEFAULT_LAEYRS = [
            HiPAttentionPerLayerConfig(
                # sliding_window_size = 777, # NOTE: debugging sw
                second_stage_k=4096,
                sa_extend_backend="streaming",
                scan_extend_backend="streaming",
                stages=_DEFAULT_STAGES,
            ),
            HiPAttentionPerLayerConfig(
                # sliding_window_size = 777, # NOTE: debugging sw
                sliding_window_size=1024,
                second_stage_k=2048,
                sa_extend_backend="streaming",
                scan_extend_backend="relative",
                stages=_DEFAULT_STAGES,
            ),
        ]

        _DEFAULT_LAEYRS_DECODE = [
            HiPAttentionPerLayerConfig(
                # sliding_window_size = 777, # NOTE: debugging sw
                second_stage_k=4096,
                sa_extend_backend="streaming",
                scan_extend_backend="streaming",
                stages=_DEFAULT_STAGES_DECODE,
            ),
            HiPAttentionPerLayerConfig(
                # sliding_window_size = 777, # NOTE: debugging sw
                second_stage_k=2048,
                sa_extend_backend="streaming",
                scan_extend_backend="relative",
                stages=_DEFAULT_STAGES_DECODE,
            ),
        ]
    else:
        _DEFAULT_LAEYRS_DECODE = _DEFAULT_LAEYRS
elif HIP_CONFIG_PRESET == "llama4":
    _DEFAULT_LAEYRS = [
        HiPAttentionPerLayerConfig(
            second_stage_k=4096,
            sa_extend_backend="streaming",
            scan_extend_backend="streaming",
        ),
        HiPAttentionPerLayerConfig(
            second_stage_k=2048,
            sa_extend_backend="streaming",
            scan_extend_backend="relative",
        ),
    ]
    _DEFAULT_STAGES_DECODE = [
        ScanStage(
            stage_block_size_q=64,
            stage_block_stride_q=1,
            stage_chunk_size=32,
            stage_k=None,
            stage_stride=1,
        ),
        ScanStage(
            stage_block_size_q=64,
            stage_block_stride_q=1,
            stage_chunk_size=16,
            stage_k=32768,
            stage_stride=1,
        ),
        ScanStage(
            stage_block_size_q=64,
            stage_block_stride_q=1,
            stage_chunk_size=4,
            stage_k=8192,
            stage_stride=1,
        ),
    ]
    _DEFAULT_LAEYRS_DECODE = [
        HiPAttentionPerLayerConfig(
            second_stage_k=4096,
            sa_extend_backend="streaming",
            scan_extend_backend="streaming",
            stages=_DEFAULT_STAGES_DECODE,
        ),
        HiPAttentionPerLayerConfig(
            second_stage_k=2048,
            sa_extend_backend="streaming",
            scan_extend_backend="relative",
            stages=_DEFAULT_STAGES_DECODE,
        ),
    ]
elif HIP_CONFIG_PRESET == "qwen3":
    _DEFAULT_LAEYRS = [
        HiPAttentionPerLayerConfig(
            second_stage_k=4096,
            sa_extend_backend="streaming",
            scan_extend_backend="streaming",
        ),
        HiPAttentionPerLayerConfig(
            second_stage_k=2048,
            sa_extend_backend="streaming",
            scan_extend_backend="relative",
        ),
    ]
    _DEFAULT_STAGES_DECODE_ST = [
        ScanStage(
            stage_block_size_q=64,
            stage_block_stride_q=1,
            stage_chunk_size=128,
            stage_k=None,
            stage_stride=1,
        ),
        ScanStage(
            stage_block_size_q=64,
            stage_block_stride_q=1,
            stage_chunk_size=16,
            stage_k=32768,
            stage_stride=1,
        ),
        ScanStage(
            stage_block_size_q=64,
            stage_block_stride_q=1,
            stage_chunk_size=4,
            stage_k=8192,
            stage_stride=1,
        ),
    ]
    _DEFAULT_STAGES_DECODE_RT = [
        ScanStage(
            stage_block_size_q=64,
            stage_block_stride_q=1,
            stage_chunk_size=32,
            stage_k=None,
            stage_stride=1,
        ),
        ScanStage(
            stage_block_size_q=64,
            stage_block_stride_q=1,
            stage_chunk_size=16,
            stage_k=32768,
            stage_stride=1,
        ),
        ScanStage(
            stage_block_size_q=64,
            stage_block_stride_q=1,
            stage_chunk_size=4,
            stage_k=8192,
            stage_stride=1,
        ),
    ]
    _DEFAULT_LAEYRS_DECODE = [
        HiPAttentionPerLayerConfig(
            sliding_window_size=24576,
            second_stage_k=4096,
            sa_extend_backend="streaming",
            scan_extend_backend="streaming",
            stages=_DEFAULT_STAGES_DECODE_ST,
        ),
        HiPAttentionPerLayerConfig(
            sliding_window_size=24576,
            second_stage_k=2048,
            sa_extend_backend="streaming",
            scan_extend_backend="relative",
            stages=_DEFAULT_STAGES_DECODE_RT,
        ),
    ]
else:
    raise Exception(f"unknown preset `{HIP_CONFIG_PRESET}`")


def try_parse_json(json_or_path: str):
    if json_or_path is None:
        parsed_json = {}
    elif isinstance(json_or_path, dict):
        parsed_json = json_or_path
    elif json_or_path.startswith("{"):
        parsed_json = json.loads(json_or_path)
    else:
        with open(json_or_path, "r") as f:
            parsed_json = json.load(f)
    return parsed_json


@dataclass
class HiPAttentionConfig:
    dense_layers: list[int] = field(
        default_factory=lambda: [
            0,
            1,
            2,
            3,
        ]
    )
    block_sparse_block_size_q: int = 64
    metadata_cache_max_batch_size: int = 32
    mask_refresh_interval: Union[int, List[int]] = field(
        default_factory=lambda: [64, 16, 8]
    )
    using_extend: bool = True
    self_extend_scale: int = 12
    layers: list[HiPAttentionPerLayerConfig] = field(
        default_factory=lambda: _DEFAULT_LAEYRS_DECODE
    )
    prefill_layers: list[HiPAttentionPerLayerConfig] = field(
        default_factory=lambda: _DEFAULT_LAEYRS
    )

    # deprecated
    apply_v_dot: bool = False
    prefill_always_dense: bool = False
    decode_always_dense: bool = False
    force_dense: bool = False
    prefill_dense_threshold: int = 8192

    json_or_path: InitVar[Optional[str]] = None
    json_override: InitVar[Optional[str]] = None

    def __post_init__(
        self,
        json_or_path: Optional[str],
        json_override: Optional[str],
    ):
        super().__init__()

        parsed_json = try_parse_json(json_or_path)
        parsed_json_override = try_parse_json(json_override)
        parsed_json.update(parsed_json_override)

        if parsed_json is not None:
            if "apply_v_dot" in parsed_json:
                self.apply_v_dot = parsed_json["apply_v_dot"]
                parsed_json.pop("apply_v_dot")
            if "dense_layers" in parsed_json:
                self.dense_layers = parsed_json["dense_layers"]
                parsed_json.pop("dense_layers")
            if "prefill_always_dense" in parsed_json:
                self.prefill_always_dense = parsed_json["prefill_always_dense"]
                parsed_json.pop("prefill_always_dense")
            if "decode_always_dense" in parsed_json:
                self.decode_always_dense = parsed_json["decode_always_dense"]
                parsed_json.pop("decode_always_dense")
            if "force_dense" in parsed_json:
                self.force_dense = parsed_json["force_dense"]
                parsed_json.pop("force_dense")
            if "prefill_dense_threshold" in parsed_json:
                self.prefill_dense_threshold = parsed_json["prefill_dense_threshold"]
                parsed_json.pop("prefill_dense_threshold")
            if "block_sparse_block_size_q" in parsed_json:
                self.block_sparse_block_size_q = parsed_json[
                    "block_sparse_block_size_q"
                ]
                parsed_json.pop("block_sparse_block_size_q")
            if "metadata_cache_max_batch_size" in parsed_json:
                self.metadata_cache_max_batch_size = parsed_json[
                    "metadata_cache_max_batch_size"
                ]
                parsed_json.pop("metadata_cache_max_batch_size")
            if "mask_refresh_interval" in parsed_json:
                assert isinstance(parsed_json["mask_refresh_interval"], (int, list))
                self.mask_refresh_interval = parsed_json["mask_refresh_interval"]
                parsed_json.pop("mask_refresh_interval")
            if "using_extend" in parsed_json:
                self.using_extend = parsed_json["using_extend"]
                parsed_json.pop("using_extend")
            if "self_extend_scale" in parsed_json:
                self.self_extend_scale = int(parsed_json["self_extend_scale"])
                parsed_json.pop("self_extend_scale")
            if "layers" in parsed_json:
                if parsed_json["layers"] is None:
                    self.layers = None
                else:
                    self.layers = [
                        HiPAttentionPerLayerConfig(parsed_json=layer)
                        for layer in parsed_json["layers"]
                    ]
                parsed_json.pop("layers")
            if "prefill_layers" in parsed_json:
                if parsed_json["prefill_layers"] is None:
                    self.prefill_layers = None
                else:
                    self.prefill_layers = [
                        HiPAttentionPerLayerConfig(parsed_json=layer)
                        for layer in parsed_json["prefill_layers"]
                    ]
                parsed_json.pop("prefill_layers")

            # FIXME following args are just temporary. need to be removed when features are stabled
            if "__delta_attention_args" in parsed_json:
                given_args = parsed_json["__delta_attention_args"]
                if os.getenv("HIP_DELTA_ATTENTION_ARGS", given_args) != given_args:
                    warnings.warn(
                        "envvar HIP_DELTA_ATTENTION_ARGS is overrided by hip attention args"
                    )
                os.environ["HIP_DELTA_ATTENTION_ARGS"] = given_args
                parsed_json.pop("__delta_attention_args")
            if "__using_dense_prefill" in parsed_json:
                given_args = parsed_json["__using_dense_prefill"]
                if os.getenv("HIP_DEBUG_USING_DENSE_PREFILL", given_args) != given_args:
                    warnings.warn(
                        "envvar HIP_DEBUG_USING_DENSE_PREFILL is overrided by hip attention args"
                    )
                os.environ["HIP_DEBUG_USING_DENSE_PREFILL"] = "1" if given_args else "0"
                parsed_json.pop("__using_dense_prefill")
            if "__head_reduce" in parsed_json:
                given_args = parsed_json["__head_reduce"]
                if os.getenv("HIP_HEAD_REDUCE", given_args) != given_args:
                    warnings.warn(
                        "envvar HIP_HEAD_REDUCE is overrided by hip attention args"
                    )
                assert int(str(given_args)) == given_args
                os.environ["HIP_HEAD_REDUCE"] = str(given_args)
                parsed_json.pop("__head_reduce")
            if "__using_landmark" in parsed_json:
                given_args = parsed_json["__using_landmark"]
                if (
                    os.getenv("HIP_DEBUG_LANDMARK_BASED_SCAN_STAGE", given_args)
                    != given_args
                ):
                    warnings.warn(
                        "envvar HIP_DEBUG_LANDMARK_BASED_SCAN_STAGE is overrided by hip attention args"
                    )
                assert (int("1" if given_args else "0") == 1) == given_args
                os.environ["HIP_DEBUG_LANDMARK_BASED_SCAN_STAGE"] = (
                    "1" if given_args else "0"
                )
                parsed_json.pop("__using_landmark")
            if "__last_dense" in parsed_json:
                given_args = parsed_json["__last_dense"]
                if os.getenv("HIP_DEBUG_LAST_DENSE", given_args) != given_args:
                    warnings.warn(
                        "envvar HIP_DEBUG_LAST_DENSE is overrided by hip attention args"
                    )
                assert int(str(given_args)) == given_args
                os.environ["HIP_DEBUG_LAST_DENSE"] = str(given_args)
                parsed_json.pop("__last_dense")
            if "__seq_thresh_fa3" in parsed_json:
                given_args = parsed_json["__seq_thresh_fa3"]
                if os.getenv("HIP_DEBUG_SEQ_THRESH_FA3", given_args) != given_args:
                    warnings.warn(
                        "envvar HIP_HEAD_REDUCE is overrided by hip attention args"
                    )
                assert int(str(given_args)) == given_args
                os.environ["HIP_DEBUG_SEQ_THRESH_FA3"] = str(given_args)
                os.environ["HIP_DEBUG_ALLOW_GATHER_KV_CACHE"] = "1"
                parsed_json.pop("__seq_thresh_fa3")

            if parsed_json:
                raise ValueError(f"Unknown keys in json: {parsed_json.keys()}")

        if (self.prefill_layers is None) and (self.layers is not None):
            self.prefill_layers = self.layers
        elif (self.prefill_layers is not None) and (self.layers is None):
            self.layers = self.prefill_layers
        elif (self.prefill_layers is None) and (self.layers is None):
            raise Exception("`prefill_layers` or `layers` should be provided")
        else:
            pass  # okay

        num_stages = len(self.layers[0].stages)
        for layer_config in self.layers:
            assert num_stages == len(layer_config.stages)

        if isinstance(self.mask_refresh_interval, int):
            self.mask_refresh_interval = [
                self.mask_refresh_interval,
            ] * num_stages

        assert (
            self.block_sparse_block_size_q
            <= self.layers[-1].stages[-1].stage_block_size_q
        )
        assert (
            self.block_sparse_block_size_q
            <= self.prefill_layers[-1].stages[-1].stage_block_size_q
        )

    def get_layer_config(self, layer_id: int, is_decode: bool):
        is_dense = layer_id in self.dense_layers

        if not is_decode:
            if len(self.prefill_layers) == 2:
                layer_config = self.prefill_layers[0 if is_dense else 1]
            else:
                layer_config = self.prefill_layers[layer_id]
        else:
            # assert dst_seq_len == 1
            if len(self.layers) == 2:
                layer_config = self.layers[0 if is_dense else 1]
            else:
                layer_config = self.layers[layer_id]

        return layer_config
