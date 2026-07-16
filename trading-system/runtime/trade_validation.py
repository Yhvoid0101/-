import math


def has_valid_stop_distances(
    entry_price: float,
    stop_distance: float,
    take_profit_distance: float,
) -> bool:
    values = (entry_price, stop_distance, take_profit_distance)
    if not all(math.isfinite(value) and value > 0 for value in values):
        return False
    return stop_distance <= entry_price * 0.5 and take_profit_distance <= entry_price


def has_auditable_signal_source(signal_source: str) -> bool:
    source = (signal_source or '').strip().lower()
    if not source:
        return False
    source = source.removeprefix('final_action=long;').removeprefix('final_action=short;')
    return source not in {'', 'none'}


def has_directionally_consistent_signal_source(action: str, signal_source: str) -> bool:
    directions = {
        part.rsplit(':', 1)[1]
        for part in (signal_source or '').lower().split(';')
        if ':' in part and part.rsplit(':', 1)[1] in {'long', 'short'}
    }
    return not directions or directions == {action}