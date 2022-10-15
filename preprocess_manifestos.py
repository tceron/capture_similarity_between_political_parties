import random
# import deepl
import pandas as pd
# from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
# from pdfminer.converter import TextConverter
# from pdfminer.layout import LAParams
# from pdfminer.pdfpage import PDFPage
from io import StringIO
import glob
from nltk import tokenize
from pathlib import Path
from itertools import combinations, product
from sklearn.model_selection import train_test_split
from collections import defaultdict
from scipy import spatial
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from utils_sim import normalize


def translate_wahl():
    df = pd.read_csv("data/Wahl-O-Mat Bundestag 2021_Datensatz_v1.02.csv")
    print(df.columns)

    texts = df["Position: Begründung"].tolist()

    translator = deepl.Translator(auth_key) #"55f63006-7d03-f4bb-e7fc-cb3301c6078a:fx"
    translated_texts=[]
    for text in texts:
        if str(text)=="nan":
            translated_texts.append(None)
        else:
            result = translator.translate_text(text, target_lang="EN-US")
            translated_text = result.text
            translated_texts.append(translated_text)

    df["justification"]=translated_texts
    print(df)
    df.to_csv("Wahl-O-Mat2021-translated.csv")


def convert_pdf_to_txt(path):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    fp = open(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos=set()

    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
        interpreter.process_page(page)

    text = retstr.getvalue()

    fp.close()
    device.close()
    retstr.close()
    f_out = open("green_2017.txt", "w")
    f_out.write(text)
    f_out.close()
    print(text)
    return text

# convert_pdf_to_txt("./party_manifestos/41113_2021.pdf")

def read_manisfesto(f):
    print(f)
    f_in = open(f, "r")
    head, doc, prev_head="", "", ""
    num_section=0
    headlines, texts, num_sections=[], [], []
    for l in f_in:
        if l.startswith(">>>>") and doc=="":
            head = l.replace("\n", " ")
            continue
        if l.startswith(">>>>") and doc!="":
            head=head.replace(">>>>", "").rstrip("\n")
            for text in tokenize.sent_tokenize(doc):
                if len(text.split(" "))>3:
                    texts.append(text)
                    headlines.append(head)
                    if head != prev_head:
                        num_section+=1
                    prev_head=head
                    num_sections.append(num_section)
            head = l
            doc=""
            continue
        else:
            l = l.replace("\n", " ").rstrip(" ").lstrip(" ")
            if not l.isdigit():
                if "CDU" in f:
                    l=l.split(" ")
                    if "-" in l[-1]:
                        tok = l[-1].split("-")[0]
                        l=" ".join(l[:-1]+[tok])
                    else:
                        l = " ".join(l[:-1])
                    if len(doc) > 2:
                        if doc.rstrip(" ")[-1]=="-":
                            doc=doc.rstrip(" ")[:-1]
                    doc+=l+" "

                else:
                    if len(doc)>2:
                        if doc[-2]=="-":
                            doc=doc[:-2]
                    doc+=l + " "
            continue
    df = pd.DataFrame(
                    {'text': texts,
                     'section': headlines,
                     'num_section': num_sections
                    })
    print(df)
    # df.to_csv("./data/manifestos_processed/" + f.split("/")[-1].replace(".txt", ".csv"))
    return df

def read_manifesto_csv(f):
    df=pd.read_csv(f)
    df = df[df['text'].notna()]
    rows = df.text.tolist()
    section_text, prev_head, head ="", "", ""
    num_section=0
    headlines, texts, num_sections = [], [], []
    for row in rows:
        if ">>>>" in row and section_text == "":
            head = row.replace(">>>>", "")
            continue
        if ">>>>" in row and section_text != "":
            section_data = tokenize.sent_tokenize(section_text)
            if head != prev_head:
                num_section += 1
            prev_head = head
            for s in section_data:
                # if len(s.split(" "))>3:
                headlines.append(head)
                texts.append(s)
                num_sections.append(num_section)
            section_text = ""
            head=row.replace(">>>>", "")
            continue
        else:
            section_text += row + " "
            continue

    df_final = pd.DataFrame(
        {'text': texts,
         'section': headlines,
         'num_section': num_sections
         })
    print(df_final)
    df_final.to_csv("./data/manifestos_processed/2013/"+f.split("/")[-1])


def dataset_finetuning():
    """
    Only if the loss is cosine similarity
    """
    files = glob.glob("./party_manifestos/2021/*.txt")
    c = 0
    rows=[]
    for f in files:
        party=f.split("/")[-1].replace(".txt", "")
        dic_used={}
        dic_headlines=read_manisfesto(f)
        headlines = list(set(dic_headlines.keys()))
        for head in headlines:
            sentences = random.sample(dic_headlines[head], len(dic_headlines[head]))
            for sent1, sent2 in combinations(sentences, 2):
                dic_used[sent1] = dic_used.get(sent1, 0)
                dic_used[sent2] = dic_used.get(sent2, 0)
                if dic_used[sent1]<=3 and dic_used[sent2]<=3:
                    row=[sent1, sent2, 1, party] #0 is the label negative because they're from the same sections
                    dic_used[sent1] = dic_used.get(sent1) + 1
                    dic_used[sent2] = dic_used.get(sent2) + 1
                    c+=1
                    rows.append(row)
        dic_used={}
        for head1, head2 in combinations(headlines, 2):
            sentences_head1 = random.sample(dic_headlines[head1], len(dic_headlines[head1]))
            sentences_head2 = random.sample(dic_headlines[head2], len(dic_headlines[head2]))
            for sent1, sent2 in product(sentences_head1, sentences_head2):
                dic_used[sent1] = dic_used.get(sent1, 0)
                dic_used[sent2] = dic_used.get(sent2, 0)
                if dic_used[sent1]<=3 and dic_used[sent2]<=3:
                    row=[sent1, sent2, 0, party] #0 is the label negative because they're from different sections
                    dic_used[sent1] = dic_used.get(sent1) + 1
                    dic_used[sent2] = dic_used.get(sent2) + 1
                    c+=1
                    rows.append(row)
    df = pd.DataFrame(rows, columns=["sent1", "sent2", "label", "party"])
    df.to_csv("./data/pairs/manifestos2021.csv")

# dataset_finetuning()

def split_val_train(df):
    first_train=pd.DataFrame()
    print(df)
    labels = set(df.num_section.tolist())
    for lab in labels:
        df_lab = df[df["num_section"]==lab]

        if len(df_lab)<10 and len(df_lab)>0:   #so that we can at least 2 in the eval set for the evaluation, otherwise it's kept only for training
            first_train = pd.concat([first_train, df_lab])
            df = df.drop(df[df.num_section == lab].index)

    print("first_train and df: ", len(first_train), len(df))
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["num_section"])
    train = pd.concat([train, first_train])
    test = create_eval_set(test)
    print(train.num_section.value_counts())
    print(test.label.value_counts())
    return train, test

def create_eval_set(df):
    print("test set:", df)
    examples = []
    labels = set(df.num_section.tolist())
    for lab in labels:
        positive = df[df["num_section"] == lab].text.tolist()
        negative = df.drop(df[df.num_section == lab].index).text.tolist()
        com=list(combinations(positive, 2))
        com_pos=random.sample(com, len(com))
        com=list(product(positive, negative))
        com_pos_neg=random.sample(com, len(com))
        for sent1, sent2 in com_pos[:int(len(com_pos)/4)]:
            examples.append([sent1, sent2, 1])

        for sent1, sent2 in com_pos_neg[:int(len(com_pos)/4)]:
            examples.append([sent1, sent2, 0])
    print("len test examples: ", len(examples))
    final_df = pd.DataFrame(examples)
    final_df.columns=["sent1", "sent2", "label"]
    return final_df


def create_dataset_claim_identifier():
    files = glob.glob("./data/original_manifestos/de/2021/*")
    df_final=pd.DataFrame()
    for f in files:
        df=pd.read_csv(f)
        df = df.dropna(subset=["cmp_code"])
        df = df[df["cmp_code"]!="H"]
        df["actor_id"]=[f.split("/")[-1].replace(".csv", "")]*len(df)
        df["label"] = [0] * len(df)
        print(df)
        df = df[["actor_id", "text", 'label', "cmp_code"]]
        df_final=pd.concat([df_final, df])
    print(df_final)
    df_final.to_csv("./data/original_manifestos/de/test2021.csv")

# create_dataset_claim_identifier()

def data_for_triplets_section(csv_files, output_dir):
    df=pd.DataFrame()
    for f in csv_files:
        tmp = pd.read_csv(f, index_col=0)
        tmp["actor_id"] = f.split("/")[-1].replace(".csv", "")
        df=pd.concat([df, tmp])
    train, test = split_val_train(df)
    train.to_csv(f"{output_dir}/train.csv")
    test.to_csv(f"{output_dir}/eval.csv")


def data_for_triplets_parties(csv_files, output_dir):
    parties = {'linke':0, 'afd':2, 'cdu':3, 'fdp':4, 'gruene':5, 'spd':6}
    df=pd.DataFrame()
    for f in csv_files:
        print(f)
        tmp = pd.read_csv(f)
        if "eu_code" in tmp.columns:
            tmp=tmp.drop(columns=["eu_code"])
        tmp = tmp[tmp["cmp_code"]!="H"].dropna(subset=["cmp_code"])
        p=f.split("/")[-1].split("_")[0].replace(".csv", "").lower()
        print(p)
        tmp["num_section"]=[parties[p]]*len(tmp)
        print(tmp.num_section.value_counts())
        df=pd.concat([df, tmp])
    train, test = split_val_train(df)
    train.to_csv(f"{output_dir}/train.csv")
    test.to_csv(f"{output_dir}/eval.csv")


def data_for_triplets_annotated_sections(csv_files, output_dir, fine_grained):
    df=pd.DataFrame()
    for f in csv_files:
        tmp = pd.read_csv(f)
        print(tmp)
        if "eu_code" in tmp.columns:
            tmp=tmp.drop(columns=["eu_code"])
        tmp = tmp[tmp["cmp_code"]!="H"].dropna(subset=["cmp_code"])
        if fine_grained:
            labels = [int(str(i)[:3])  if i != 0.0 else int("000") for i in tmp.cmp_code.tolist()]
        else:
            labels = [int(str(i)[:1]) for i in tmp.cmp_code.tolist()]
        tmp["num_section"]=labels
        df=pd.concat([df, tmp])
    train, test = split_val_train(df)
    train.to_csv(f"{output_dir}/train.csv")
    test.to_csv(f"{output_dir}/eval.csv")


# for year in ["2017-2021", "2021-2013"]:
#     csv_files = []
#     if "-" in year:
#         ye = year.split("-")
#         for y in ye:
#             csv_files.extend(glob.glob(f"./data/original_manifestos/de/{y}/*.csv"))
#     else:
#         csv_files.extend(glob.glob(f"./data/original_manifestos/de/{year}/*.csv"))
#     output_dir=f"./data/triplets/annotated_sections/{year}"
#     Path(f"{output_dir}").mkdir(exist_ok=True, parents=True)
#     data_for_triplets_annotated_sections(csv_files, output_dir, False)
    # output_dir=f"./data/triplets/by_party/{year}"
    # Path(f"{output_dir}").mkdir(exist_ok=True, parents=True)
    # data_for_triplets_parties(csv_files, output_dir)

df=pd.read_csv("./data/Wahl-O-Mat2021-translated.csv", index_col=0)
df = df[df["Partei: Nr."].isin([1, 2,3,4, 5, 6])]
print(df["Position: Position"].value_counts())


# df_claims = pd.read_csv("./data/predicted_claims.csv", index_col=0).claim_quote.tolist()
# true_claims = set(df[df["label"]==1].text.tolist())
# true_positives=[]
# for i in true_claims:
#     if i in df_claims:
#         true_positives.append(i)
# print(len(true_positives))

# df = pd.read_csv("./data/test.csv")
# print(df.actor_id.value_counts())



