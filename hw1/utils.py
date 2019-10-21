def ksubset(K:int,lst:list):
    """
    generate K subset of a list
    generate function
    """
    def subset(cur:list,idx:int):
        if len(cur) == K:
            # print(cur)
            yield tuple(cur)
        else:
            if idx < len(lst) and (len(cur) + len(lst) - idx >= K):
                yield from subset(cur + [lst[idx]],idx+1)
                yield from subset(cur , idx+1)
    
    yield from subset([],0)

def membership_test(x : set,c:tuple):
        for e in c:
            if e not in x:
                return False
        return True

def generatek_subset(lst: tuple):
    if len(lst) == 2:
        return [(lst[0],lst[1]),(lst[1],lst[0])]
    return [
        (lst[0:i]+lst[i+1 : ], lst[i])
        for i in range(len(lst))
    ]

