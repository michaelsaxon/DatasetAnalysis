# DatasetAnalysis

Research project on dealing with broken/biased datasets for NLP

## Setup instructions (Linux server)


### Python dependencies

Set up a personal Anaconda installation on the NLP servers. Then:

```
conda create -n DSAnalysis python=3.8
```

Set this env up for Pytorch

```
conda activate DSAnalysis
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

Install the requirements.txt:

```
pip install -r requirements.txt
```

Finally, configure Weights & Biases

```
wandb login
```

and paste in either Michael's **private key** or use your own.

### Setting up working directory, env vars, data

Currently, the scripts hard code for where data is located. Michael will fix this very soon.

The data is located at `nlp.cs.ucsb.edu:/mnt/hdd/saxon/{dataset}` for datasets `snli_1.0`, `anli_v1.0`, `mnli`. Can access directly from there or copy.

You will want to set up a working directory where trained models, precomp'd embeddings can be saved.

## TODOs

- [x] complete basic implementation
- [x] Finish and check requirements.txt
- [x] Implement working dir stuff as 
- [x] modularized PECO scoring
- [x] side-by-side model/condition performance (How much recovery is actually recovery)
- [x] FEVER
- [x] s1only
- [x] Loading from a checkpoint in train_classifier.py
- [X] counterfactualNLI
- [x] XNLI
- [ ] implement MNLI separate train/test for each split
- [ ] Using gradient to find word-level salience in decisions (to further identify cheating)
- [ ] comparing gradient salience importance between the two models
- [ ] Binary label for a decision: is the salience in S1 or S2 
- [ ] aNLI (for future work)
- [ ] modularized graph analysis
- [ ] PECO scoring for lesioned (instead of S2only) (which might matter)
- [ ] Frequency list computation for levels of n-gram
- [ ] Dataset filtration code (x domain changing)
- [ ] Compute entropy for dataset, classes, etc
- [ ] Corruption code (x domain)
- [ ] Corruption code (y domain)
