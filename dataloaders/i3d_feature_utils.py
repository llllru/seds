import os

import numpy as np
import torch


I3D_PARTS = ("left", "right", "face", "body")


def resolve_i3d_feature_path(root, subset, video_name, part):
    if not root:
        return None

    candidates = [
        os.path.join(root, subset, "{}_{}.npy".format(video_name, part)),
        os.path.join(root, "{}_{}.npy".format(video_name, part)),
        os.path.join(root, subset, "{}.npy".format(video_name)),
        os.path.join(root, "{}.npy".format(video_name)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def load_i3d_feature_array(root, subset, video_name, part):
    path = resolve_i3d_feature_path(root, subset, video_name, part)
    if path is None:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError("Missing {} I3D feature: {}".format(part, path))
    feature = np.load(path).astype(np.float32)
    if feature.ndim != 2:
        raise ValueError("{} I3D feature must be [T, D], got {}".format(part, feature.shape))
    return feature


def load_i3d_feature_dict(roots, subset, video_name):
    features = {}
    for part, root in roots.items():
        if root:
            features[part] = load_i3d_feature_array(root, subset, video_name, part)
    return features


def valid_pose_clip_count(sample):
    mask = sample["body"]["pose_mask"]
    return int(torch.sum(mask[1:] == 0).item())


def _set_pose_clip_count(part_sample, clip_count):
    mask = part_sample["pose_mask"]
    clips_start = part_sample["clips_start"]
    mask[1:] = 1
    mask[1 : clip_count + 1] = 0
    clips_start[clip_count:] = -1


def align_sample_with_i3d_features(sample, features, feature_len):
    if not features:
        return sample

    lengths = [valid_pose_clip_count(sample)]
    lengths.extend(feature.shape[0] for feature in features.values())
    clip_count = min(lengths + [feature_len])

    for part in ("right", "left", "body"):
        _set_pose_clip_count(sample[part], clip_count)

    dim = next(iter(features.values())).shape[1]
    i3d = {}
    for part, feature in features.items():
        if feature.shape[1] != dim:
            raise ValueError("All I3D features for one sample must share dim {}".format(dim))
        i3d[part] = torch.from_numpy(feature[:clip_count]).float()

    i3d_mask = torch.zeros(clip_count + 1, dtype=torch.long)
    sample["i3d"] = i3d
    sample["i3d_mask"] = i3d_mask
    return sample


def collate_i3d_features(batch):
    if not batch or "i3d" not in batch[0]:
        return {}

    parts = sorted(batch[0]["i3d"].keys())
    collated = {}
    max_len = max(sample["i3d"][parts[0]].shape[0] for sample in batch)
    for part in parts:
        dim = batch[0]["i3d"][part].shape[1]
        padded_features = []
        for sample in batch:
            feature = sample["i3d"][part]
            padded = torch.zeros(max_len, dim, dtype=torch.float32)
            padded[: feature.shape[0]] = feature.float()
            padded_features.append(padded)
        collated["{}_i3d".format(part)] = torch.stack(padded_features, dim=0).float()

    padded_masks = []
    for sample in batch:
        clip_len = sample["i3d"][parts[0]].shape[0]
        mask = torch.ones(max_len + 1, dtype=torch.long)
        mask[: clip_len + 1] = 0
        padded_masks.append(mask)
    collated["i3d_mask"] = torch.stack(padded_masks, dim=0).long()
    return collated


def get_i3d_feature_roots(args):
    if not getattr(args, "use_i3d_local_features", False):
        return {}
    return {
        "left": getattr(args, "i3d_left_features_path", ""),
        "right": getattr(args, "i3d_right_features_path", ""),
        "face": getattr(args, "i3d_face_features_path", ""),
        "body": getattr(args, "i3d_body_features_path", ""),
    }
