def traceback_all(seq, D, l=1, delta=0):
    """Zwraca posortowana liste struktur dot-bracket z >= max - delta par."""
    n = len(seq)
    s = " " + seq

    def val(i, j):
        return D[i][j] if j >= i else 0

    @lru_cache(maxsize=None)
    def tb(i, j, p):
        """Wszystkie struktury na [i, j] z DOKLADNIE p parami (krotka stringow)."""
        if j < i:
            return ("",) if p == 0 else ()
        if p > val(i, j) or p < 0:
            return ()
        res = []
        # j sparowane z k -- k malejaco (kolejnosc jak na stronie)
        for k in range(j - l - 1, i - 1, -1):
            if (s[k], s[j]) not in PAIRS:
                continue
            for a in range(0, val(i, k - 1) + 1):
                b = p - 1 - a
                if b < 0 or b > val(k + 1, j - 1):
                    continue
                for ls in tb(i, k - 1, a):
                    for ins in tb(k + 1, j - 1, b):
                        res.append(ls + "(" + ins + ")")
        # j niesparowane -- na koncu
        for st in tb(i, j - 1, p):
            res.append(st + ".")
        return tuple(res)

    mx = D[1][n]
    out = []
    for p in range(mx, max(mx - delta, 0) - 1, -1):
        out.extend(tb(1, n, p))
    return out

structs = traceback_all(seq, D, a.l, a.delta)
shown = structs[: a.n] if a.n > 0 else structs
for st in shown:
    print(st)
