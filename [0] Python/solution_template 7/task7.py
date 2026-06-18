def find_modified_max_argmax(L,f):
    L = [f(x)for x in L if type(x)==int]
    return L and (m:=max(L),L.index(m))or()
