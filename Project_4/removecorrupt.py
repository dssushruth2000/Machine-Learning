import os, glob
files = glob.glob("Monkey Species Data/*/*/*")
for file in files :
   f = open(file, "rb") # open to read binary file
   if not b"JFIF" in f.peek(10) :
          f.close()
          os.remove(file)
   else :
          f.close()