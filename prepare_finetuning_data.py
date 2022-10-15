import pandas as pd
import glob
from pathlib import Path
from itertools import combinations, product
from sklearn.model_selection import train_test_split
import random

def split_val_train(df):
    first_train=pd.DataFrame()
    labels = set(df.label.tolist())
    for lab in labels:
        df_lab = df[df["label"]==lab]

        if len(df_lab)<10 and len(df_lab)>0:   #so that we can at least 2 in the eval set for the evaluation, otherwise it's kept only for training
            first_train = pd.concat([first_train, df_lab])
            df = df.drop(df[df.label == lab].index)

    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    train = pd.concat([train, first_train])
    test = create_eval_set(test)
    print(train.label.value_counts())
    print(test.label.value_counts())
    return train, test


def create_eval_set(df):
    examples = []
    labels = set(df.label.tolist())
    for lab in labels:
        positive = df[df["label"] == lab].text.tolist()
        negative = df.drop(df[df.label == lab].index).text.tolist()
        com=list(combinations(positive, 2))
        com_pos=random.sample(com, len(com))
        com=list(product(positive, negative))
        com_pos_neg=random.sample(com, len(com))
        for sent1, sent2 in com_pos[:int(len(com_pos)/4)]:
            examples.append([sent1, sent2, 1])

        for sent1, sent2 in com_pos_neg[:int(len(com_pos)/4)]:
            examples.append([sent1, sent2, 0])
    final_df = pd.DataFrame(examples)
    final_df.columns=["sent1", "sent2", "label"]
    return final_df


def triplets_annotated_sections(csv_files, output_dir):
    df=pd.DataFrame()
    for f in csv_files:
        tmp = pd.read_csv(f)
        tmp.columns = ["text", "label", "0"]
        tmp = tmp[tmp["label"]!="H"].dropna(subset=["label"])

        labels = [int(str(i)[:1]) for i in tmp.label.tolist()]

        tmp["label"]=labels
        df=pd.concat([df, tmp])
    df = df[["text", "label"]]
    train, test = split_val_train(df)
    train.to_csv(f"{output_dir}/train.csv")
    test.to_csv(f"{output_dir}/eval.csv")


def triplets_parties(csv_files, output_dir):
    parties = sorted(set([i.split("/")[-1].replace(".csv", "") for i in csv_files]))
    parties = {p:n for n, p in enumerate(parties)}
    print(parties)
    df=pd.DataFrame()
    for f in csv_files:
        tmp = pd.read_csv(f)
        p=f.split("/")[-1].split("_")[0].replace(".csv", "").lower()
        tmp["label"]=[parties[p]]*len(tmp)
        df=pd.concat([df, tmp])
    df = df[["text", "label"]]
    train, test = split_val_train(df)
    train.to_csv(f"{output_dir}/train.csv")
    test.to_csv(f"{output_dir}/eval.csv")


def prepare_data(years):
    csv_files = []
    if "-" in years:
        ye = years.split("-")
        for y in ye:
            csv_files.extend(glob.glob(f"./data/manifestos/{y}/*.csv"))
    else:
        csv_files.extend(glob.glob(f"./data/manifestos/{years}/*.csv"))
    output_dir=f"./data/triplets/annotated_sections/{years}"
    Path(f"{output_dir}").mkdir(exist_ok=True, parents=True)
    triplets_annotated_sections(csv_files, output_dir)
    output_dir=f"./data/triplets/by_party/{years}"
    Path(f"{output_dir}").mkdir(exist_ok=True, parents=True)
    triplets_parties(csv_files, output_dir)
