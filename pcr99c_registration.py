"""Optimized PyTorch implementation of PCR99c point-cloud registration."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import torch

Tensor = torch.Tensor


@dataclass
class RegistrationResult:
    rotation: Tensor  # [3, 3]
    translation: Tensor  # [3]
    inlier_mask: Tensor  # [N], bool
    num_inliers: int
    evaluated_hypotheses: int
    success: bool

    def __iter__(self):
        """Allow ``R, t = pcr99c(...)`` while retaining diagnostics."""
        yield self.rotation
        yield self.translation


def _matlab_points_to_rows(points: Tensor, name: str) -> Tensor:
    """Accept MATLAB [3,N], or unambiguous Python [N,3], point storage."""
    if points.ndim != 2:
        raise ValueError(f"{name} must be a 2-D tensor")
    if points.shape[0] == 3:
        # In the ambiguous 3x3 case, retain the MATLAB convention: columns are
        # points. Internal CLI code bypasses this helper and always uses [N,3].
        return points.transpose(0, 1).contiguous()
    if points.shape[1] == 3:
        return points.contiguous()
    raise ValueError(f"{name} must have shape [3,N] or [N,3]")

def _rt_from_3points_batched(
    source: Tensor, target: Tensor, scale: float
) -> Tuple[Tensor, Tensor, Tensor]:
    """Vectorized MATLAB Rt_from_3points for [H,3,3] row-wise triplets."""
    centroid_source = source.mean(dim=1)
    centroid_target = target.mean(dim=1)

    src_v12 = source[:, 1] - source[:, 0]
    src_v13 = source[:, 2] - source[:, 0]
    tgt_v12 = target[:, 1] - target[:, 0]
    tgt_v13 = target[:, 2] - target[:, 0]

    src_x_norm = torch.norm(src_v12, dim=1)
    tgt_x_norm = torch.norm(tgt_v12, dim=1)
    src_y_raw = torch.cross(src_v12, src_v13, dim=1)
    tgt_y_raw = torch.cross(tgt_v12, tgt_v13, dim=1)
    src_y_norm = torch.norm(src_y_raw, dim=1)
    tgt_y_norm = torch.norm(tgt_y_raw, dim=1)

    valid = (
        (src_x_norm > 0)
        & (tgt_x_norm > 0)
        & (src_y_norm > 0)
        & (tgt_y_norm > 0)
        & torch.isfinite(src_x_norm)
        & torch.isfinite(tgt_x_norm)
        & torch.isfinite(src_y_norm)
        & torch.isfinite(tgt_y_norm)
    )

    tiny = torch.finfo(source.dtype).tiny
    src_x = src_v12 / src_x_norm.clamp_min(tiny).unsqueeze(1)
    tgt_x = tgt_v12 / tgt_x_norm.clamp_min(tiny).unsqueeze(1)
    src_y = src_y_raw / src_y_norm.clamp_min(tiny).unsqueeze(1)
    tgt_y = tgt_y_raw / tgt_y_norm.clamp_min(tiny).unsqueeze(1)

    src_z_raw = torch.cross(src_x, src_y, dim=1)
    tgt_z_raw = torch.cross(tgt_x, tgt_y, dim=1)
    src_z_norm = torch.norm(src_z_raw, dim=1)
    tgt_z_norm = torch.norm(tgt_z_raw, dim=1)
    valid &= (
        (src_z_norm > 0)
        & (tgt_z_norm > 0)
        & torch.isfinite(src_z_norm)
        & torch.isfinite(tgt_z_norm)
    )

    src_z = src_z_raw / src_z_norm.clamp_min(tiny).unsqueeze(1)
    tgt_z = tgt_z_raw / tgt_z_norm.clamp_min(tiny).unsqueeze(1)

    source_basis = torch.stack((src_x, src_y, src_z), dim=2)
    target_basis = torch.stack((tgt_x, tgt_y, tgt_z), dim=2)
    rotation = torch.bmm(target_basis, source_basis.transpose(1, 2))
    translation = centroid_target - scale * torch.bmm(
        rotation, centroid_source.unsqueeze(2)
    ).squeeze(2)
    valid &= torch.isfinite(rotation).all(dim=2).all(dim=1)
    valid &= torch.isfinite(translation).all(dim=1)
    return rotation, translation, valid

def rt_from_3points(source: Tensor, target: Tensor, scale: float = 1.0):
    """Translate MATLAB Rt_from_3points; inputs use MATLAB [3,3] layout."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    source_rows = _matlab_points_to_rows(source, "source")
    target_rows = _matlab_points_to_rows(target, "target")
    if source_rows.shape != (3, 3) or target_rows.shape != (3, 3):
        raise ValueError("Rt_from_3points requires exactly three point pairs")
    rotations, translations, valid = _rt_from_3points_batched(
        source_rows.unsqueeze(0), target_rows.unsqueeze(0), scale
    )
    if not bool(valid[0]):
        nan_rotation = torch.full_like(rotations[0], float("nan"))
        nan_translation = torch.full_like(translations[0], float("nan"))
        return nan_rotation, nan_translation
    return rotations[0], translations[0]

def _svd(matrix: Tensor) -> Tuple[Tensor, Tensor]:
    """Return U,V for matrix=U*S*V' on both modern and older PyTorch."""
    if hasattr(torch, "linalg") and hasattr(torch.linalg, "svd"):
        u, _, vh = torch.linalg.svd(matrix, full_matrices=False)
        return u, vh.transpose(0, 1)
    u, _, v = torch.svd(matrix)
    return u, v

def _rt_from_n_points_rows(
    source: Tensor, target: Tensor, scale: float
) -> Tuple[Tensor, Tensor]:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must have matching shape [N,3]")
    if source.shape[0] < 3:
        raise ValueError("at least three point pairs are required")

    centroid_source = source.mean(dim=0)
    centroid_target = target.mean(dim=0)
    source_centered = source - centroid_source
    target_centered = target - centroid_target

    # MATLAB: [UU,~,VV] = svd(A*B'); R=VV*D*UU'.  With row-wise
    # storage, source_centered.T @ target_centered is the same A*B'.
    covariance = source_centered.transpose(0, 1).mm(target_centered)
    u, v = _svd(covariance)
    v_corrected = v.clone()
    if float(torch.det(v.mm(u.transpose(0, 1))).item()) < 0.0:
        v_corrected[:, -1].neg_()
    rotation = v_corrected.mm(u.transpose(0, 1))
    translation = centroid_target - scale * rotation.mv(centroid_source)
    return rotation, translation

def rt_from_n_points(source: Tensor, target: Tensor, scale: float = 1.0):
    """Translate MATLAB Rt_from_N_points for [3,N] (or unambiguous [N,3])."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    source_rows = _matlab_points_to_rows(source, "source")
    target_rows = _matlab_points_to_rows(target, "target")
    return _rt_from_n_points_rows(source_rows, target_rows, scale)

def _stable_argsort(values: Tensor) -> Tensor:
    try:
        return torch.argsort(values, stable=True)
    except TypeError:  # PyTorch versions predating the stable keyword.
        return torch.argsort(values)

def _rank_correspondences(
    source: Tensor,
    target: Tensor,
    scale: float,
    thr1: float,
    block_size: int,
) -> Tuple[Tensor, Tensor]:
    """Compute MATLAB ranking costs and its Boolean ``large_error_mat``.

    Pairwise distances are still formed in blocks so temporary floating-point
    storage stays bounded.  The retained N-by-N matrix uses one byte per entry
    (``torch.bool``), and candidate-triplet screening later becomes three
    indexed lookups, matching the MATLAB implementation.
    """
    n = source.shape[0]
    costs = torch.zeros(n, dtype=source.dtype, device=source.device)
    large_error_mat = torch.empty(
        (n, n), dtype=torch.bool, device=source.device
    )
    source_squared_norm = (source * source).sum(dim=1)
    target_squared_norm = (target * target).sum(dim=1)
    log_scale = math.log(scale)

    for row_start in range(0, n, block_size):
        row_end = min(row_start + block_size, n)
        source_rows = source[row_start:row_end]
        target_rows = target[row_start:row_end]

        for col_start in range(row_start, n, block_size):
            col_end = min(col_start + block_size, n)
            source_cols = source[col_start:col_end]
            target_cols = target[col_start:col_end]

            source_distance2 = (
                source_squared_norm[row_start:row_end, None]
                + source_squared_norm[None, col_start:col_end]
            )
            source_distance2.addmm_(
                source_rows,
                source_cols.transpose(0, 1),
                beta=1.0,
                alpha=-2.0,
            ).clamp_min_(0.0)

            target_distance2 = (
                target_squared_norm[row_start:row_end, None]
                + target_squared_norm[None, col_start:col_end]
            )
            target_distance2.addmm_(
                target_rows,
                target_cols.transpose(0, 1),
                beta=1.0,
                alpha=-2.0,
            ).clamp_min_(0.0)

            same_block = row_start == col_start
            if same_block:
                source_distance2.diagonal().zero_()
                target_distance2.diagonal().zero_()

            # 0.5*log(d_est^2/d_gt^2) is log(d_est/d_gt), as in MATLAB.
            pair_error = torch.log(target_distance2)
            pair_error.sub_(torch.log(source_distance2)).mul_(0.5)
            pair_error.sub_(log_scale).abs_()

            # MATLAB: large_error_bool = r > thr1.  In particular, NaN > thr1
            # is false while infinity > thr1 is true.
            large_error_block = pair_error > thr1
            if same_block:
                large_error_block.diagonal().zero_()
                large_error_mat[
                    row_start:row_end, col_start:col_end
                ].copy_(large_error_block)
            else:
                large_error_mat[
                    row_start:row_end, col_start:col_end
                ].copy_(large_error_block)
                large_error_mat[
                    col_start:col_end, row_start:row_end
                ].copy_(large_error_block.transpose(0, 1))

            # MATLAB nansum ignores 0/0 edges. One-sided zero distances give
            # infinity and are truncated to thr1.
            pair_error = torch.where(
                torch.isnan(pair_error), torch.zeros_like(pair_error), pair_error
            )
            pair_error = torch.where(
                torch.isinf(pair_error),
                torch.full_like(pair_error, thr1),
                pair_error,
            )
            pair_error.clamp_max_(thr1)
            if same_block:
                pair_error.diagonal().zero_()

            costs[row_start:row_end].add_(pair_error.sum(dim=1))
            if not same_block:
                costs[col_start:col_end].add_(pair_error.sum(dim=0))

    return _stable_argsort(costs), large_error_mat

def _rank_sum_triplet_chunks(n: int, chunk_size: int) -> Iterator[Tensor]:
    """Yield i<j<k in the exact rank-sum order of the MATLAB nested loops."""
    buffer = torch.empty((chunk_size, 3), dtype=torch.long)
    used = 0

    # MATLAB: for s = 6 : n+(n-1)+(n-2), using one-based indices.
    for index_sum in range(6, 3 * n - 2):
        i_min = max(1, index_sum - 2 * n + 1)
        i_max = (index_sum - 3) // 3

        for i in range(i_min, i_max + 1):
            j_min = max(i + 1, index_sum - i - n)
            j_max = (index_sum - i - 1) // 2
            next_j = j_min

            while next_j <= j_max:
                take = min(j_max - next_j + 1, chunk_size - used)
                destination = buffer[used : used + take]
                j_values = torch.arange(next_j, next_j + take, dtype=torch.long)
                destination[:, 0].fill_(i - 1)
                destination[:, 1].copy_(j_values - 1)
                destination[:, 2].copy_(index_sum - i - j_values - 1)
                used += take
                next_j += take

                if used == chunk_size:
                    yield buffer.clone()
                    used = 0

    if used:
        yield buffer[:used].clone()

def _compatible_triplets(
    large_error_mat: Tensor,
    ranked_to_original: Tensor,
    triplets: Tensor,
) -> Tensor:
    """Apply MATLAB's prescreening using three Boolean matrix lookups."""
    original_triplets = ranked_to_original[triplets]
    first = original_triplets[:, 0]
    second = original_triplets[:, 1]
    third = original_triplets[:, 2]
    return ~(
        large_error_mat[first, second]
        | large_error_mat[second, third]
        | large_error_mat[third, first]
    )

def _score_point_chunk_size(
    num_hypotheses: int,
    num_points: int,
    dtype: torch.dtype,
    score_memory_mb: float,
) -> int:
    element_size = torch.empty((), dtype=dtype).element_size()
    memory_bytes = max(1, int(score_memory_mb * 1024 * 1024))
    # Scoring holds approximately two [H,point_chunk] floating tensors.
    max_elements = max(1, memory_bytes // (2 * element_size))
    return max(1, min(num_points, max_elements // max(1, num_hypotheses)))

def _score_hypotheses(
    source: Tensor,
    target: Tensor,
    rotations: Tensor,
    translations: Tensor,
    scale: float,
    squared_inlier_threshold: float,
    score_memory_mb: float,
) -> Tensor:
    num_hypotheses = rotations.shape[0]
    num_points = source.shape[0]
    point_chunk = _score_point_chunk_size(
        num_hypotheses, num_points, source.dtype, score_memory_mb
    )
    counts = torch.zeros(
        num_hypotheses, dtype=torch.long, device=source.device
    )

    for start in range(0, num_points, point_chunk):
        end = min(start + point_chunk, num_points)
        source_t = source[start:end].transpose(0, 1)
        residual_squared: Optional[Tensor] = None

        # Three Hx3 by 3xP matrix products avoid materializing an Hx3xP
        # transformed-point tensor. Peak memory is about two HxP tensors.
        for coordinate in range(3):
            coordinate_residual = rotations[:, coordinate, :].mm(source_t)
            coordinate_residual.mul_(scale)
            coordinate_residual.add_(translations[:, coordinate].unsqueeze(1))
            coordinate_residual.sub_(
                target[start:end, coordinate].unsqueeze(0)
            ).square_()
            if residual_squared is None:
                residual_squared = coordinate_residual
            else:
                residual_squared.add_(coordinate_residual)

        assert residual_squared is not None
        counts.add_((residual_squared <= squared_inlier_threshold).sum(dim=1))

    return counts

def _residual_squared(
    source: Tensor,
    target: Tensor,
    rotation: Tensor,
    translation: Tensor,
    scale: float,
) -> Tensor:
    predicted = scale * source.mm(rotation.transpose(0, 1))
    predicted.add_(translation.unsqueeze(0))
    residual = target - predicted
    return (residual * residual).sum(dim=1)

def _evaluate_triplet_batch(
    source: Tensor,
    target: Tensor,
    triplets: Tensor,
    scale: float,
    squared_inlier_threshold: float,
    score_memory_mb: float,
) -> Tuple[int, Optional[Tensor], Optional[Tensor]]:
    triplet_source = source[triplets]
    triplet_target = target[triplets]
    rotations, translations, valid = _rt_from_3points_batched(
        triplet_source, triplet_target, scale
    )

    valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        return 0, None, None

    valid_rotations = rotations[valid_indices]
    valid_translations = translations[valid_indices]
    counts = _score_hypotheses(
        source,
        target,
        valid_rotations,
        valid_translations,
        scale,
        squared_inlier_threshold,
        score_memory_mb,
    )
    best_valid_index = int(torch.argmax(counts).item())
    best_count = int(counts[best_valid_index].item())
    return (
        best_count,
        valid_rotations[best_valid_index],
        valid_translations[best_valid_index],
    )

def _failure_result(n: int, reference: Tensor, evaluated: int) -> RegistrationResult:
    rotation = torch.full(
        (3, 3), float("nan"), dtype=reference.dtype, device=reference.device
    )
    translation = torch.full(
        (3,), float("nan"), dtype=reference.dtype, device=reference.device
    )
    inlier_mask = torch.zeros(n, dtype=torch.bool, device=reference.device)
    return RegistrationResult(rotation, translation, inlier_mask, 0, evaluated, False)

def register_points(
    source: Tensor,
    target: Tensor,
    sigma: float,
    thr1: float,
    thr2: float,
    n_hypo: int,
    scale: float,
    ranking_block_size: int,
    triplet_chunk_size: int,
    score_memory_mb: float,
    max_hypotheses: int,
    early_stop_ratio: float,
    early_stop_min: int,
    disable_early_stop: bool,
) -> RegistrationResult:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must have matching shape [N,3]")
    n = source.shape[0]
    if n < 3:
        return _failure_result(n, source, 0)
    if not source.is_floating_point() or source.dtype != target.dtype:
        raise ValueError("source and target must use the same floating dtype")
    if source.device != target.device:
        raise ValueError("source and target must be on the same device")
    if sigma <= 0 or thr1 <= 0 or thr2 <= 0 or scale <= 0:
        raise ValueError("sigma, thr1, thr2, and scale must be positive")
    if n_hypo <= 0 or ranking_block_size <= 0 or triplet_chunk_size <= 0:
        raise ValueError("batch and block sizes must be positive")
    if score_memory_mb <= 0 or max_hypotheses < 0:
        raise ValueError("score memory must be positive and max_hypotheses nonnegative")

    sort_indices, large_error_mat = _rank_correspondences(
        source, target, scale, thr1, ranking_block_size
    )
    ranked_source = source[sort_indices].contiguous()
    ranked_target = target[sort_indices].contiguous()

    squared_inlier_threshold = (sigma * thr2) ** 2
    early_stop_count: Optional[int]
    if disable_early_stop:
        early_stop_count = None
    else:
        # Positive-number equivalent of MATLAB round(n*0.009).
        ratio_count = int(math.floor(n * early_stop_ratio + 0.5))
        early_stop_count = max(early_stop_min, ratio_count)

    best_count = 0
    best_rotation: Optional[Tensor] = None
    best_translation: Optional[Tensor] = None
    evaluated = 0
    pending = torch.empty((0, 3), dtype=torch.long, device=source.device)
    stop = False

    def evaluate(batch: Tensor) -> None:
        nonlocal best_count, best_rotation, best_translation, evaluated
        evaluated += int(batch.shape[0])
        count, rotation, translation = _evaluate_triplet_batch(
            ranked_source,
            ranked_target,
            batch,
            scale,
            squared_inlier_threshold,
            score_memory_mb,
        )
        # Preserve MATLAB's strict improvement test; the first hypothesis in a
        # tied batch is selected by torch.argmax, matching MATLAB max.
        if count > best_count and rotation is not None and translation is not None:
            best_count = count
            best_rotation = rotation.clone()
            best_translation = translation.clone()

    for raw_triplets_cpu in _rank_sum_triplet_chunks(n, triplet_chunk_size):
        raw_triplets = raw_triplets_cpu.to(source.device, non_blocking=True)
        compatible = _compatible_triplets(
            large_error_mat, sort_indices, raw_triplets
        )
        valid_triplets = raw_triplets[compatible]
        if valid_triplets.numel() == 0:
            continue

        pending = torch.cat((pending, valid_triplets), dim=0)
        while pending.shape[0] > 0:
            remaining_limit = (
                max_hypotheses - evaluated if max_hypotheses > 0 else n_hypo
            )
            if max_hypotheses > 0 and remaining_limit <= 0:
                stop = True
                break

            current_batch_size = min(n_hypo, remaining_limit)
            if pending.shape[0] < current_batch_size:
                break

            evaluate(pending[:current_batch_size])
            pending = pending[current_batch_size:].clone()

            if early_stop_count is not None and best_count >= early_stop_count:
                stop = True
                break
            if max_hypotheses > 0 and evaluated >= max_hypotheses:
                stop = True
                break

        if stop:
            break

    # The MATLAB code silently discarded a final incomplete hypothesis batch.
    # Evaluating it can only preserve or improve the selected model.
    if not stop and pending.shape[0] > 0:
        remaining_limit = (
            max_hypotheses - evaluated if max_hypotheses > 0 else pending.shape[0]
        )
        take = min(int(pending.shape[0]), int(remaining_limit))
        if take > 0:
            evaluate(pending[:take])

    if best_rotation is None or best_translation is None:
        return _failure_result(n, source, evaluated)

    initial_residual2 = _residual_squared(
        ranked_source, ranked_target, best_rotation, best_translation, scale
    )
    ranked_inlier_mask = initial_residual2 <= squared_inlier_threshold
    num_inliers = int(ranked_inlier_mask.sum().item())
    if num_inliers < 3:
        return _failure_result(n, source, evaluated)

    final_rotation, final_translation = _rt_from_n_points_rows(
        ranked_source[ranked_inlier_mask],
        ranked_target[ranked_inlier_mask],
        scale,
    )

    # Return the mask in the original TXT correspondence order.
    original_inlier_mask = torch.zeros(
        n, dtype=torch.bool, device=source.device
    )
    original_inlier_mask[sort_indices] = ranked_inlier_mask
    return RegistrationResult(
        final_rotation,
        final_translation,
        original_inlier_mask,
        num_inliers,
        evaluated,
        True,
    )

def pcr99c(
    xyz_gt: Tensor,
    xyz_est: Tensor,
    sigma: float,
    thr1: float,
    thr2: float,
    n_hypo: int,
    scale: float = 1.0,
    *,
    ranking_block_size: int = 1024,
    triplet_chunk_size: int = 65536,
    score_memory_mb: float = 256.0,
    max_hypotheses: int = 0,
    early_stop_ratio: float = 0.009,
    early_stop_min: int = 9,
    disable_early_stop: bool = False,
) -> RegistrationResult:
    """PyTorch PCR99c.

    ``xyz_gt`` and ``xyz_est`` follow the MATLAB function's [3,N] convention.
    An unambiguous [N,3] tensor is also accepted. The returned result can be
    unpacked as ``R, t = pcr99c(...)``.
    """
    source = _matlab_points_to_rows(xyz_gt, "xyz_gt")
    target = _matlab_points_to_rows(xyz_est, "xyz_est")
    return register_points(
        source,
        target,
        sigma,
        thr1,
        thr2,
        n_hypo,
        scale,
        ranking_block_size,
        triplet_chunk_size,
        score_memory_mb,
        max_hypotheses,
        early_stop_ratio,
        early_stop_min,
        disable_early_stop,
    )
