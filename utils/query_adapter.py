import torch
from torch import Tensor


def _time_scalar(mark: Tensor) -> Tensor:
    if mark.dim() == 3:
        return mark[..., 0]
    return mark


def flatten_apn_batch(
    x: Tensor,
    x_mark: Tensor,
    x_mask: Tensor,
    y: Tensor | None,
    y_mark: Tensor,
    y_mask: Tensor,
) -> dict[str, Tensor]:
    """
    Convert APN-style dense padded batches into padded query-set tensors.

    Padding entries have mask value 0. Query target values are zero-filled when
    y is None, which is used at test time to avoid feeding future values to the
    denoiser.
    """
    batch_size = x.shape[0]
    device = x.device
    value_dtype = x.dtype

    x_time = _time_scalar(x_mark)
    y_time = _time_scalar(y_mark)
    x_valid = x_mask > 0
    y_valid = y_mask > 0

    hist_counts = x_valid.flatten(1).sum(dim=1).long()
    query_counts = y_valid.flatten(1).sum(dim=1).long()
    hist_max = max(int(hist_counts.max().item()) if batch_size else 0, 1)
    query_max = max(int(query_counts.max().item()) if batch_size else 0, 1)

    hist_value = torch.zeros(batch_size, hist_max, device=device, dtype=value_dtype)
    hist_time = torch.zeros(batch_size, hist_max, device=device, dtype=x_time.dtype)
    hist_var = torch.zeros(batch_size, hist_max, device=device, dtype=torch.long)
    hist_mask = torch.zeros(batch_size, hist_max, device=device, dtype=value_dtype)

    query_value = torch.zeros(batch_size, query_max, device=device, dtype=value_dtype)
    query_time = torch.zeros(batch_size, query_max, device=device, dtype=y_time.dtype)
    query_var = torch.zeros(batch_size, query_max, device=device, dtype=torch.long)
    query_row = torch.zeros(batch_size, query_max, device=device, dtype=torch.long)
    query_mask = torch.zeros(batch_size, query_max, device=device, dtype=value_dtype)

    for b in range(batch_size):
        hist_rows, hist_cols = torch.nonzero(x_valid[b], as_tuple=True)
        n_hist = hist_rows.numel()
        if n_hist > 0:
            hist_value[b, :n_hist] = x[b, hist_rows, hist_cols]
            hist_time[b, :n_hist] = x_time[b, hist_rows]
            hist_var[b, :n_hist] = hist_cols.long()
            hist_mask[b, :n_hist] = 1

        query_rows, query_cols = torch.nonzero(y_valid[b], as_tuple=True)
        n_query = query_rows.numel()
        if n_query > 0:
            if y is not None:
                query_value[b, :n_query] = y[b, query_rows, query_cols]
            query_time[b, :n_query] = y_time[b, query_rows]
            query_var[b, :n_query] = query_cols.long()
            query_row[b, :n_query] = query_rows.long()
            query_mask[b, :n_query] = 1

    return {
        "hist_value": hist_value,
        "hist_time": hist_time,
        "hist_var": hist_var,
        "hist_mask": hist_mask,
        "query_value": query_value,
        "query_time": query_time,
        "query_var": query_var,
        "query_row": query_row,
        "query_mask": query_mask,
    }


def scatter_query_values(
    query_values: Tensor,
    query_row: Tensor,
    query_var: Tensor,
    query_mask: Tensor,
    output_shape: tuple[int, int, int],
) -> Tensor:
    """Scatter flattened query values back to APN dense [B, Ly, C] format."""
    dense = torch.zeros(output_shape, device=query_values.device, dtype=query_values.dtype)
    batch_size = query_values.shape[0]
    for b in range(batch_size):
        valid = query_mask[b] > 0
        if valid.any():
            dense[b, query_row[b, valid], query_var[b, valid]] = query_values[b, valid]
    return dense


def scatter_query_samples(
    query_samples: Tensor,
    query_row: Tensor,
    query_var: Tensor,
    query_mask: Tensor,
    output_shape: tuple[int, int, int],
) -> Tensor:
    """Scatter sparse samples [B, N, K] to dense [B, N, Ly, C] format."""
    batch_size, num_samples, _ = query_samples.shape
    _, output_length, num_variables = output_shape
    dense = torch.zeros(
        batch_size,
        num_samples,
        output_length,
        num_variables,
        device=query_samples.device,
        dtype=query_samples.dtype,
    )
    for b in range(batch_size):
        valid = query_mask[b] > 0
        if valid.any():
            dense[b, :, query_row[b, valid], query_var[b, valid]] = query_samples[
                b, :, valid
            ]
    return dense


def masked_mse(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    residual = (pred - target) * mask
    return (residual ** 2).sum() / mask.sum().clamp_min(1.0)


def masked_mae(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    residual = (pred - target) * mask
    return residual.abs().sum() / mask.sum().clamp_min(1.0)
