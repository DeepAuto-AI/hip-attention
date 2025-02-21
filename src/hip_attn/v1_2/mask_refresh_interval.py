from hip_attn.v1_2.hip_config import HiPAttentionConfig


class HiPMaskRefreshState:
    def __init__(self, hip_config: HiPAttentionConfig):
        self.hip_config = hip_config
        self.decode_index = 0

    def update(self):
        metadata_cached_stages = None

        if self.hip_config.mask_refresh_interval is not None:
            require_refresh = False

            for i_stage, refresh_inteval in enumerate(
                self.hip_config.mask_refresh_interval
            ):
                if self.decode_index % refresh_inteval == 0 and not require_refresh:
                    metadata_cached_stages = i_stage
                    require_refresh = True

            if not require_refresh:
                metadata_cached_stages = None

        if self.decode_index == 0:
            metadata_cached_stages = -1

        self.decode_index += 1

        return metadata_cached_stages
