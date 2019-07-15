import os, sys
import shutil
import random
from shutil import copyfile

for dirname in os.listdir('./mini'):
    print (dirname)
    try:
        os.makedirs('./test/' + dirname)
    except:
        print ("blah")
    onefile = random.sample(os.listdir('./mini/' + dirname), 1)
    copyfile("./mini/" + dirname + "/" + onefile[0], "./test/" + dirname + "/" + onefile[0])

