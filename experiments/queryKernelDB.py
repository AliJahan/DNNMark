#! /usr/bin/env python

import sqlite3
import sys
import os.path
import collections

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(BASE_DIR, "alexnet_kernel.db")
conn = sqlite3.connect(db_path)

table_name = 'alexnet_kernel_dict'

c = conn.cursor()

if len(sys.argv) < 3:
  print "batch size and arch need to be specified!!!"
  exit()

batchsize = sys.argv[1]
arch = sys.argv[2]

c.execute("SELECT DISTINCT kernelname FROM "+table_name+" WHERE batchsize = "+batchsize+" and arch = '"+arch+"'")
kernel_name_list = c.fetchall()
for kernel in kernel_name_list:
  c.execute("select invocationorder from "+table_name+" where kernelname = '"+kernel[0]+"' and batchsize = "+batchsize+" and arch = '"+arch+"'")
  max_invocation_count = max(c.fetchall())[0]
  print kernel[0]+","+str(max_invocation_count)

