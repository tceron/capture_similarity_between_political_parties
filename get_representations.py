from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

import fasttext
import gzip
import wget
import shutil

import pandas as pd
import torch
from pathlib import Path
import pickle
import numpy as np
from nltk.corpus import stopwords
import nltk
from sklearn.preprocessing import LabelEncoder
nltk.download('stopwords')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sbert_representations(sentences, type_model):
    print(f"Loading {type_model}...")
    model = SentenceTransformer(type_model)
    embeddings = model.encode(sentences, convert_to_tensor=True).cpu().detach().numpy()

    return embeddings


def bert_representations(sentences, type_model):
    print(f"Loading {type_model}...")
    model = AutoModel.from_pretrained(type_model, return_dict=True, output_hidden_states=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(type_model)

    embeddings = []
    for i in range(len(sentences)):
        inputs = tokenizer.encode_plus(sentences[i], return_tensors='pt').to(device)
        outputs = model(**inputs)

        layer_output = torch.stack(outputs.hidden_states[-2:]).sum(0).squeeze().cpu().detach().numpy()

        embeddings.append(np.mean(np.vstack(layer_output[1:-1, :]), axis=0))

    return embeddings

def fasttext_representations(sentences):
    url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.de.300.bin.gz"
    wget.download(url)
    with gzip.open(url.split("/")[-1], 'rb') as f_in:
        with open(url.split("/")[-1].replace(".gz", ""), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    print(f"Loading fasttext model...")
    model = fasttext.load_model(url.split("/")[-1].replace(".gz", ""))

    german_stop_words = stopwords.words('german')
    embeddings=[]
    for sent in sentences:
        sent=fasttext.tokenize(sent)
        tokens = [t for t in sent if t not in german_stop_words] #CountVectorizer(stop_words=german_stop_words)
        embs =  np.vstack([model[t] for t in tokens]).mean(0)
        embeddings.append(embs)
    return embeddings


def get_embeddings(df, output_dir, type_model=None):
    df = df.dropna()
    le = LabelEncoder()
    df.label = le.fit_transform(np.array([str(i)[0] for i in df.label.tolist()]))

    parties = np.array(df["party"].tolist())
    sentences = df["text"].tolist()
    categories = df.label.tolist()

    if "bert" in type_model:
        embeddings = bert_representations(sentences, type_model)
    elif "fasttext" in type_model:
        embeddings = fasttext_representations(sentences)
    else:
        embeddings = sbert_representations(sentences, type_model)

    embeddings = np.c_[np.vstack(embeddings), parties]
    embeddings = np.c_[embeddings, categories]
    print(embeddings.shape)

    pickle.dump(embeddings, open(f"{output_dir}/embeddings.p", "wb"))


def get_representations_datasets(finetuning_year, train=False):

    outdir = "embeddings"
    models = ["fasttext_emb", "paraphrase-multilingual-mpnet-base-v2",
              "xlm-roberta-base", "bert-base-german-cased"]
    if train:
        models.extend([f"./outputs/fine_tuned/by_domain/{finetuning_year}",
                      f"./outputs/fine_tuned/by_party/{finetuning_year}"])
    else:
        models.extend(["tceron/sentence-transformers-party-similarity-by-domain",
                       "tceron/sentence-transformers-party-similarity-by-party"])

    for dataset in ["manifesto", "claim_identifier"]:  #, manifesto] corresponds to all sentences
        if dataset=="claim_identifier":
            df = pd.read_csv("./data/predicted_claims_2021.csv", index_col=0)
        else:
            df = pd.read_csv("./data/manifestos.csv", index_col=0)
        print(dataset, "len dataset: ", len(df))

        for m in models:

            if "outputs" in m:
                name_m = m.split("/")[-2]
                output_dir = f"./{outdir}/{name_m}/{finetuning_year}/{dataset}"
            elif "tceron" in m:
                name_m = m.split("/")[-1]
                output_dir = f"./{outdir}/{name_m}/{finetuning_year}/{dataset}"
            else:
                output_dir = f"./{outdir}/{m}/{finetuning_year}/{dataset}"
            print(output_dir)
            Path(output_dir).mkdir(exist_ok=True, parents=True)

            get_embeddings(df, output_dir, m)

