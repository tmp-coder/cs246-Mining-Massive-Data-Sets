"""
data_clean for p2
"""

from pyspark import SparkConf,SparkContext


# data Clean

DATA_IN = 'graph-full.txt'

DEG_OUT = 'deg_out' # format : (u,deg_out,out_nodes)
DEG_IN = 'deg_in' # format: (u,in_nodes)

def main():

    conf = SparkConf()
    sc = SparkContext(conf=conf)
    import os,shutil

    if os.path.exists(DEG_IN):
        shutil.rmtree(DEG_IN)
    if os.path.exists(DEG_OUT):
        shutil.rmtree(DEG_OUT)


    text_lines = sc.textFile(DATA_IN)

    def clean(xlst):
        se = set(xlst)
        return len(se),list(se)
    text_lines.map(
        lambda x : tuple(map(lambda e : int(e)-1,x.split('\t')))
    ).groupByKey().mapValues(clean).saveAsPickleFile(DEG_OUT)
    sc.textFile(DATA_IN).map(
        lambda x : tuple(map(lambda e : int(e)-1,reversed(x.split('\t'))))
    ).groupByKey().mapValues(lambda x: list(set(x))).saveAsPickleFile(DEG_IN)

if __name__ == "__main__":
    main()    


