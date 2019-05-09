import os 

for path, dirs, files in os.walk('.'):
  print(path)
  for f in files:
    print(f)