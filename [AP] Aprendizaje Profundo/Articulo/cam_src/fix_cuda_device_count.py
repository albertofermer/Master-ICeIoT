import torch
from typing import Set
import os
import re
import torch
from torch.cuda import _raw_device_count_nvml


def _parse_visible_devices() -> Set[int | str]:
    """Parse CUDA_VISIBLE_DEVICES environment variable."""
    var = os.getenv("CUDA_VISIBLE_DEVICES")
    if var is None:
        return set(x for x in range(64))

    def decode_id(s: str) -> int | str:
        """Return -1 or positive integer sequence string starts with,"""
        if not s:
            return -1
        s = s.strip()

        try:
            return int(s)
        except ValueError:
            pass

        if re.match(r'GPU-.+', s) is not None:
            return s

        return -1

    # CUDA_VISIBLE_DEVICES uses something like strtoul
    # which makes `1gpu2,2ampere` is equivalent to `1,2`
    rc: Set[int | str] = set()
    for elem in var.split(","):
        rc.add(decode_id(elem.strip()))
    return rc

def _device_count_nvml() -> int:
    """Return number of devices as reported by NVML taking CUDA_VISIBLE_DEVICES into account.
    Negative value is returned if NVML discovery or initialization has failed."""
    visible_devices = _parse_visible_devices()
    if not visible_devices:
        return 0

    try:
        raw_cnt = _raw_device_count_nvml()
    except OSError:
        return -1
    except AttributeError:
        return -1
    
    if raw_cnt <= 0:
        return raw_cnt
    return min(raw_cnt, len(visible_devices))


torch.cuda._device_count_nvml = _device_count_nvml
torch.cuda._parse_visible_devices = _parse_visible_devices