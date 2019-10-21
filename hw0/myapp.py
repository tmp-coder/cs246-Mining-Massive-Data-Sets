import sys
import numpy as np


import pyspark as psk

conf = psk.SparkConf()
sc = psk.SparkContext(conf=conf)
print ("%d lines" % sc.textFile(sys.argv[1]).count())