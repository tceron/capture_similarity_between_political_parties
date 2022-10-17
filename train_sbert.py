from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses, evaluation
from torch.utils.data import DataLoader
import pandas as pd
import torch
from pathlib import Path

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_examples(folder, is_eval=False):
    examples = []
    if not is_eval:
        df = pd.read_csv(f"./data/{folder}/train.csv", index_col=0)
        print(df)
        l = df.values.tolist()
        for i in range(len(l)):
            examples.append(InputExample(texts=[l[i][0]], label=int(l[i][1])))
        return examples
    else:
        df = pd.read_csv(f"./data/{folder}/eval.csv", index_col=0)
        print(df)
        l = df.values.tolist()
        for row in l:
            examples.append(InputExample(texts=[row[0], row[1]], label=row[2]))
        print("len examples: ", len(examples))
        return examples


def train(data_dir, output_dir):

    Path(output_dir).mkdir(exist_ok=True, parents=True)
    train_examples = load_examples(data_dir)
    #Define the model. Either from scratch of by loading a pre-trained model
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

    #Define your train dataset, the dataloader and the train loss
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.BatchAllTripletLoss(model)

    eval_examples = load_examples(data_dir, is_eval=True)
    evaluator = evaluation.BinaryClassificationEvaluator.from_input_examples(eval_examples, name='eval')

    model.fit(train_objectives=[(train_dataloader, train_loss)],
                                evaluator=evaluator,
                                epochs=5,
                                warmup_steps=100,
                                evaluation_steps=100,
                                save_best_model=True,
                                output_path=output_dir)  #evaluator=evaluator,


def eval(data_dir, output_dir, model=True):
    if not model:
        model = SentenceTransformer(output_dir)
    eval_examples = load_examples(data_dir, is_eval=True)

    evaluator = evaluation.BinaryClassificationEvaluator.from_input_examples(eval_examples, name='evaluation')
    evaluator(model, output_path=output_dir)


def run_trainsbert(years):
    folders = [ "by_party", "by_domain"]
    print(folders)

    for folder in folders:
        output_dir = f"./outputs/fine_tuned/{folder}/{years}"
        print(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        data_dir = f"triplets/{folder}/{years}"
        train(data_dir, output_dir)
        eval(data_dir, output_dir, False)