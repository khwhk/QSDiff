import torch
from torch import Tensor


def _time_scalar(mark: Tensor) -> Tensor:
    return mark[..., 0] if mark.dim() == 3 else mark


def _bin_indices(times: Tensor, start: Tensor, end: Tensor, n_bins: int) -> Tensor:
    if n_bins <= 1 or float((end - start).abs()) < 1e-8:
        return torch.zeros_like(times, dtype=torch.long)
    position = (times - start) / (end - start).clamp_min(1e-8)
    return torch.round(position.clamp(0, 1) * (n_bins - 1)).long()


def _grid_times(start: Tensor, end: Tensor, n_bins: int) -> Tensor:
    if n_bins <= 1:
        return start.reshape(1)
    weight = torch.linspace(0, 1, n_bins, device=start.device, dtype=start.dtype)
    return start + weight * (end - start)


def _delta_since_observation(mask: Tensor) -> Tensor:
    """Return normalized time since the last observation for [T, C] masks."""
    time_length, n_variables = mask.shape
    delta = torch.ones_like(mask)
    last_seen = torch.full((n_variables,), -1, device=mask.device, dtype=torch.long)
    scale = max(time_length - 1, 1)
    for time_index in range(time_length):
        observed = mask[time_index] > 0
        last_seen = torch.where(observed, torch.full_like(last_seen, time_index), last_seen)
        elapsed = torch.where(last_seen >= 0, time_index - last_seen, scale)
        delta[time_index] = elapsed.to(mask.dtype) / scale
    return delta


def gridify_irregular_batch(
    x: Tensor,
    x_mark: Tensor,
    x_mask: Tensor,
    y: Tensor,
    y_mark: Tensor,
    y_mask: Tensor,
    history_bins: int,
    future_bins: int,
) -> dict[str, Tensor]:
    """Map padded irregular events to a fixed grid without using future values as input."""
    batch_size, _, n_variables = x.shape
    x_time = _time_scalar(x_mark)
    y_time = _time_scalar(y_mark)
    dtype = x.dtype
    device = x.device

    history_value = torch.zeros(batch_size, history_bins, n_variables, device=device, dtype=dtype)
    history_mask = torch.zeros_like(history_value)
    history_time = torch.zeros(batch_size, history_bins, device=device, dtype=x_time.dtype)
    future_value = torch.zeros(batch_size, future_bins, n_variables, device=device, dtype=dtype)
    future_mask = torch.zeros_like(future_value)
    future_time = torch.zeros(batch_size, future_bins, device=device, dtype=y_time.dtype)
    query_bin = torch.zeros(batch_size, y.shape[1], device=device, dtype=torch.long)

    for batch_index in range(batch_size):
        history_rows = x_mask[batch_index].any(dim=-1)
        query_rows = y_mask[batch_index].any(dim=-1)

        valid_history_time = x_time[batch_index, history_rows]
        valid_query_time = y_time[batch_index, query_rows]
        if valid_history_time.numel() == 0:
            history_start = x_time.new_tensor(0.0)
            history_end = x_time.new_tensor(1.0)
        else:
            history_start = valid_history_time.min()
            history_end = valid_history_time.max()
        if valid_query_time.numel() == 0:
            future_start = history_end
            future_end = history_end + 1
        else:
            future_start = valid_query_time.min()
            future_end = valid_query_time.max()

        history_time[batch_index] = _grid_times(history_start, history_end, history_bins)
        future_time[batch_index] = _grid_times(future_start, future_end, future_bins)

        history_index = _bin_indices(x_time[batch_index], history_start, history_end, history_bins)
        future_index = _bin_indices(y_time[batch_index], future_start, future_end, future_bins)
        query_bin[batch_index] = future_index

        history_sum = torch.zeros_like(history_value[batch_index])
        history_count = torch.zeros_like(history_value[batch_index])
        future_sum = torch.zeros_like(future_value[batch_index])
        future_count = torch.zeros_like(future_value[batch_index])

        for variable in range(n_variables):
            observed_history = x_mask[batch_index, :, variable] > 0
            if observed_history.any():
                bins = history_index[observed_history]
                values = x[batch_index, observed_history, variable]
                history_sum[:, variable].scatter_add_(0, bins, values)
                history_count[:, variable].scatter_add_(0, bins, torch.ones_like(values))

            observed_future = y_mask[batch_index, :, variable] > 0
            if observed_future.any():
                bins = future_index[observed_future]
                values = y[batch_index, observed_future, variable]
                future_sum[:, variable].scatter_add_(0, bins, values)
                future_count[:, variable].scatter_add_(0, bins, torch.ones_like(values))

        history_mask[batch_index] = (history_count > 0).to(dtype)
        future_mask[batch_index] = (future_count > 0).to(dtype)
        history_value[batch_index] = history_sum / history_count.clamp_min(1)
        future_value[batch_index] = future_sum / future_count.clamp_min(1)

    history_delta = torch.stack(
        [_delta_since_observation(mask) for mask in history_mask], dim=0
    )
    return {
        "history_value": history_value,
        "history_mask": history_mask,
        "history_delta": history_delta,
        "history_time": history_time,
        "future_value": future_value,
        "future_mask": future_mask,
        "future_time": future_time,
        "query_bin": query_bin,
    }


def gridify_future_queries(
    y: Tensor,
    y_mark: Tensor,
    y_mask: Tensor,
    future_bins: int,
) -> dict[str, Tensor]:
    """Map only future query targets to a fixed grid.

    This is the matched target-representation control used by QSDiff: history
    remains in its original irregular form, while future query values are
    averaged within fixed temporal bins and accompanied by an observation mask.
    ``query_bin`` maps predictions on that grid back to the original queries.
    """
    if future_bins <= 0:
        raise ValueError(f"future_bins must be positive, got {future_bins}")

    batch_size, _, n_variables = y.shape
    y_time = _time_scalar(y_mark)
    dtype = y.dtype
    device = y.device

    future_value = torch.zeros(
        batch_size, future_bins, n_variables, device=device, dtype=dtype
    )
    future_mask = torch.zeros_like(future_value)
    future_time = torch.zeros(
        batch_size, future_bins, device=device, dtype=y_time.dtype
    )
    query_bin = torch.zeros(
        batch_size, y.shape[1], device=device, dtype=torch.long
    )

    for batch_index in range(batch_size):
        query_rows = y_mask[batch_index].any(dim=-1)
        valid_query_time = y_time[batch_index, query_rows]
        if valid_query_time.numel() == 0:
            future_start = y_time.new_tensor(0.0)
            future_end = y_time.new_tensor(1.0)
        else:
            future_start = valid_query_time.min()
            future_end = valid_query_time.max()

        future_time[batch_index] = _grid_times(
            future_start, future_end, future_bins
        )
        future_index = _bin_indices(
            y_time[batch_index], future_start, future_end, future_bins
        )
        query_bin[batch_index] = future_index

        future_sum = torch.zeros_like(future_value[batch_index])
        future_count = torch.zeros_like(future_value[batch_index])
        for variable in range(n_variables):
            observed = y_mask[batch_index, :, variable] > 0
            if observed.any():
                bins = future_index[observed]
                values = y[batch_index, observed, variable]
                future_sum[:, variable].scatter_add_(0, bins, values)
                future_count[:, variable].scatter_add_(
                    0, bins, torch.ones_like(values)
                )

        future_mask[batch_index] = (future_count > 0).to(dtype)
        future_value[batch_index] = future_sum / future_count.clamp_min(1)

    return {
        "future_value": future_value,
        "future_mask": future_mask,
        "future_time": future_time,
        "query_bin": query_bin,
    }


def gather_grid_queries(grid_values: Tensor, query_bin: Tensor, output_mask: Tensor) -> Tensor:
    """Gather [B, T, C] grid values back to padded irregular query rows."""
    gather_index = query_bin.unsqueeze(-1).expand(-1, -1, grid_values.shape[-1])
    return torch.gather(grid_values, dim=1, index=gather_index) * output_mask
