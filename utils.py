

# Credits: http://www.locallyoptimal.com/blog/2013/01/20/elegant-n-gram-generation-in-python/
def generate_ngrams(input_list, n):
    return list(zip(*[input_list[i:] for i in range(n)]))


def jaccard(a, b):
    return len(a.intersection(b)) / len(a.union(b))

