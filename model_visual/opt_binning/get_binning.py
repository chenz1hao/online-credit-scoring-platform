from optbinning import OptimalBinning


## https://github.com/guillermo-navas-palencia/optbinning
def binning(feature, x, y):
    optb = OptimalBinning(name=feature, dtype="numerical", solver="cp")
    for i in range(len(y)):
        if y[i] == -1:
            y[i] = 0
    optb.fit(x, y)
    return optb.splits

