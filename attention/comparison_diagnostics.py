"""Named non-operational diagnostics shared by the A-line comparators."""
SHADOW_FIELDS = frozenset({'shadow_C2_converging'})


def shadow_differences(left, right, path=''):
    """Keep each changed shadow value visible without relaxing working gates."""
    rows = []
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(left.keys() | right.keys()):
            here = path + '/' + key
            if key in SHADOW_FIELDS:
                if key not in left or key not in right or left[key] != right[key]:
                    rows.append(dict(path=here, left=left.get(key), right=right.get(key),
                                     left_present=key in left, right_present=key in right))
            elif key in left and key in right:
                rows.extend(shadow_differences(left[key], right[key], here))
    elif isinstance(left, list) and isinstance(right, list):
        for i, (a, b) in enumerate(zip(left, right)):
            rows.extend(shadow_differences(a, b, path + '/' + str(i)))
    return rows
