import tarfile

original_path = "./hakatoon-salamat"
tgz_path = original_path + ".tar.gz"

with tarfile.open(tgz_path, "w:gz") as tar:
   tar.add(original_path)