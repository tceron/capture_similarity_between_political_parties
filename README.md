## Capture similarity between political parties

The code available in this repository corresponds to our work on optimizing text representation to capture similarity between parties based on their manifestos. 

More info on: Tanise Ceron, Nico Blokker, and Sebastian Pado. 2022. Optimizing text representations to capture (dis)similarity between political parties. _Conference on Computational Natural Language Learning (CoNLL)_, Abu Dhabi, United Arab Emirates.

### Paper's abstract:
Even though fine-tuned neural language models have been pivotal in enabling “deep” automatic text analysis, optimizing text representations for specific applications remains a crucial bottleneck.  In this study, we look at this problem in the context of a task from computational social science, namely modeling pairwise similarities between political parties. Our research question is what level of structural information is necessary to create robust text representation, contrasting a strongly informed approach (which uses both claim span and claim category annotations) with approaches that forgo one or both types of annotation with document structure-based heuristics. Evaluating our models on the manifestos of German parties for the 2021 federal election. We find that heuristics that maximize within-party over between-party similarity along with a normalization step lead to reliable party similarity prediction, without the need for manual annotation.


## Code
First create a new environment and install the necessary packages:

    python3 -m venv venv
    
    source venv/bin/activate

    pip install -r requirements.txt

If you'd like to train SBERT from scratch, run:

    python3 run.py --train

However, bear in mind that the results of the SBERTpart and SBERTdomain models won't yield the same results as in the paper. You can reproduce the same results by loading the models used in the paper from the HuggingFace Hub, for that run:

    python3 run.py


### Citations

Our work can be cited as: 

@inproceedings{ceron2022,
 author = {Ceron, Tanise and Blokker, Nico and Pad{\'o} Sebastian},
 booktitle = {Proceedings of the 26th Conference on Computational Natural Language Learning},
 title = {Optimizing text representations to capture (dis)similarity between political parties},
 url = {},
 volume = {},
 year = {2022}
}


If you're using the manifesto dataset found in this paper, make sure to cite: 

@article{manifesto, title = {Manifesto Corpus. Version: 2021.1}, author = {Burst, Tobias and Krause, Werner and Lehmann, Pola and
Lewandowski, Jirka and Matthieß, Theres and Merz, Nicolas and Regel, Sven and Zehnter, Lisa}, year = {2021}, journal = {Berlin: WZB Berlin Social Science Center.} }

