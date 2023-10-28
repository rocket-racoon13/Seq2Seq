import pandas as pd

df = pd.read_csv("data/opendict-korean-proverb.csv", encoding="cp949", header=None)
texts = []
for i in range(len(df)):
    row = df.iloc[i]
    texts.append(row.values[0])
    
with open("data/opendict-korean-proverb.txt", "w", encoding="utf-8-sig") as f_out:
    f_out.write("\n".join(texts))