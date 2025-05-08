import zipfile

with zipfile.ZipFile("submission.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
    zipf.write("main.py")
    zipf.write("mydata.pkl.xz")
    zipf.write("MCTS_sumbit.py")

print("submission.zip created successfully.")
