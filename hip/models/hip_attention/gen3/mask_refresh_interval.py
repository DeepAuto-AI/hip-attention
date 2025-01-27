from hip.models.hip_attention.gen3.hip_config import HiPAttentionConfig


class HiPMaskRefreshState:
    def __init__(self):
        self.decode_index = 0

    def update(self, is_decode, is_extend, hip_config: HiPAttentionConfig):
        metadata_cached_stages = None

        if is_decode:
            if hip_config.mask_refresh_interval is not None:
                require_refresh = False

                for i_stage, refresh_inteval in enumerate(hip_config.mask_refresh_interval):
                    if self.decode_index % refresh_inteval == 0 and not require_refresh:
                        metadata_cached_stages = i_stage
                        require_refresh = True

                if not require_refresh:
                    metadata_cached_stages = None

            self.decode_index += 1

        elif is_extend:
            self.decode_index = 0

        return metadata_cached_stages
