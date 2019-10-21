import re
import sys
import os,shutil
import pyspark as psk

FILE_IN = sys.argv[1]
FILE_OUT = sys.argv[2]

if os.path.isdir(FILE_OUT):
    shutil.rmtree(FILE_OUT)


conf = psk.SparkConf()
sc = psk.SparkContext(conf=conf)
pat = r'[\s|\d]+'
lines = sc.textFile(FILE_IN)

maps = lines.flatMap(lambda line:
 [(word[0].lower(),1) for word in filter(lambda word: len(word)>0 and word[0].lower() >='a' and word[0].lower()<='z',
                                    re.split(pat,line)) ])

count = maps.reduceByKey(lambda n1,n2: n1+n2)

count.saveAsTextFile(FILE_OUT)
sc.stop()
