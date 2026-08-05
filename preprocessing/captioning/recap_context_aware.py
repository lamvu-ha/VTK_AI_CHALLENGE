"""
Context-aware recap: sinh caption có tham chiếu ngữ cảnh các keyframe trước.
Giữ tính nhất quán trong cùng một shot.
"""
from typing import List, Dict, Optional
from preprocessing.captioning.vlm_caption_keyframe import VLMCaptionRunner


class RecapContextAware:
    """
    Sinh caption cho keyframe với context từ các frame trước (trong cùng shot).
    """
    def __init__(self, vlm_runner: VLMCaptionRunner, context_window: int = 2):
        self.vlm = vlm_runner
        self.context_window = context_window

    def caption_with_context(self, keyframes: List[Dict]) -> List[Dict]:
        """
        Với mỗi keyframe, dùng caption của context_window frame trước làm ngữ cảnh.
        """
        results = []
        captions_so_far = []

        for i, kf in enumerate(keyframes):
            # Lấy context từ các caption trước
            ctx_caps = captions_so_far[-self.context_window:]
            if ctx_caps:
                context_str = " | ".join(ctx_caps)
                prompt = (
                    f"Context from previous frames: {context_str}\n"
                    f"Now briefly describe this new image, keeping consistency with the context."
                )
            else:
                prompt = "Describe this image briefly in English."

            cap = self.vlm.caption(kf.get("path", ""), prompt=prompt)
            captions_so_far.append(cap)
            results.append({**kf, "caption": cap, "has_context": bool(ctx_caps)})

        return results

    def batch_recap_by_shot(self, keyframes: List[Dict]) -> List[Dict]:
        """
        Nhóm keyframe theo shot_idx, recap từng shot riêng.
        """
        from collections import defaultdict
        shot_groups: dict = defaultdict(list)
        for kf in keyframes:
            shot_groups[kf.get("shot_idx", 0)].append(kf)

        all_results = []
        for shot_idx in sorted(shot_groups):
            shot_kfs = shot_groups[shot_idx]
            all_results.extend(self.caption_with_context(shot_kfs))
        return all_results
