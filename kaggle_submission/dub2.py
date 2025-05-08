import zipfile

with zipfile.ZipFile("submission.zip", "r") as z:
    print("📦 Zip 파일 안의 내용:")
    z.printdir()
