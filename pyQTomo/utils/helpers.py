import itertools as it

def make_lexigraphical_mask(n):
    """
    Make an index sorting mask for something that is in reverse lexigraphical
    order into lexigraphical order
    """
    lexi = list(it.product(range(2), repeat=n))
    lexi_reverse = [val[::-1] for val in lexi]
    map = []
    for i, non_lexi in enumerate(lexi_reverse):
        for j, yes_lexi in enumerate(lexi):
            if non_lexi == yes_lexi:
                map.append(j)
    return map   