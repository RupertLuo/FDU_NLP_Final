import configparser
import json
import spacy
from spacy.matcher import Matcher
import sys
import timeit
from tqdm import tqdm
import numpy as np
import pandas as pd
blacklist = {"-PRON-", "actually", "likely", "possibly", "want", "make", "my", "someone", "sometimes_people",
             "sometimes", "would", "want_to", "one", "something", "sometimes", "everybody", "somebody", "could",
             "could_be"}


concept_vocab = set()
config = configparser.ConfigParser()
config.read("paths.cfg")
with open(config["paths"]["concept_vocab"], "r", encoding="utf8") as f:
    cpnet_vocab = [l.strip() for l in list(f.readlines())]
cpnet_vocab = [c.replace("_", " ") for c in cpnet_vocab]

def lemmatize(nlp, concept):

    doc = nlp(concept.replace("_"," "))
    lcs = set()
    lcs.add("_".join([token.lemma_ for token in doc])) # all lemma
    return lcs

def load_matcher(nlp):
    config = configparser.ConfigParser()
    config.read("paths.cfg")
    with open(config["paths"]["matcher_patterns"], "r", encoding="utf8") as f:
        all_patterns = json.load(f)

    matcher = Matcher(nlp.vocab)
    for concept, pattern in tqdm(all_patterns.items(), desc="Adding patterns to Matcher."):
        matcher.add(concept, None, pattern)
    return matcher

def ground_mentioned_concepts(nlp, matcher, s, ans = ""):
    s = s.lower()
    doc = nlp(s)
    matches = matcher(doc)

    mentioned_concepts = set()
    span_to_concepts = {}

    for match_id, start, end in matches:

        span = doc[start:end].text  # the matched span
        if len(set(span.split(" ")).intersection(set(ans.split(" ")))) > 0:
            continue
        original_concept = nlp.vocab.strings[match_id]
        # print("Matched '" + span + "' to the rule '" + string_id)

        if len(original_concept.split("_")) == 1:
            original_concept = list(lemmatize(nlp, original_concept))[0]

        if span not in span_to_concepts:
            span_to_concepts[span] = set()

        span_to_concepts[span].add(original_concept)

    for span, concepts in span_to_concepts.items():
        concepts_sorted = list(concepts)
        concepts_sorted.sort(key=len)

        # mentioned_concepts.update(concepts_sorted[0:2])

        shortest = concepts_sorted[0:3] #
        for c in shortest:
            if c in blacklist:
                continue
            lcs = lemmatize(nlp, c)
            intersect = lcs.intersection(shortest)
            if len(intersect)>0:
                mentioned_concepts.add(list(intersect)[0])
            else:
                mentioned_concepts.add(c)
    return mentioned_concepts

def hard_ground(nlp, sent):
    global cpnet_vocab
    sent = sent.lower()
    doc = nlp(sent)
    res = set()
    for t in doc:
        if t.lemma_ in cpnet_vocab:
            res.add(t.lemma_)
    sent = "_".join([t.text for t in doc])
    if sent in cpnet_vocab:
        res.add(sent)
    return res

def match_mentioned_concepts(nlp, sent1, sent2, batch_id = -1):
    matcher = load_matcher(nlp)

    res = []
    # print("Begin matching concepts.")
    for sid, s in tqdm(enumerate(sent1), total=len(sent1), desc="grounding batch_id:%d"%batch_id):
        a = sent2[sid]
        sent1_concepts = ground_mentioned_concepts(nlp, matcher, s)
        sent2_concepts = ground_mentioned_concepts(nlp, matcher, a)
        if len(sent1)==0:
            # print(s)
            sent1_concepts = hard_ground(nlp, s) # not very possible
            print(sent1_concepts)
        if len(sent2)==0:
            print(a)
            sent2_concepts = hard_ground(nlp, a) # some case
            print(sent2_concepts)

        res.append({"sent1": list(sent1_concepts), "sent2": list(sent2_concepts)})
    return res

def mua():
    data = pd.read_csv('../data/train_w.tsv', delimiter='\t', header=None)
    sent1 = list(data.iloc[:,3])
    sent2 = list(data.iloc[:,4])
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
    nlp.add_pipe(nlp.create_pipe('sentencizer'))

    res = match_mentioned_concepts(nlp, sent1=sent1, sent2=sent2)
    sent1s_concept = ['***'.join(pair["sent1"]) for pair in res]
    sent2s_concept = ['***'.join(pair["sent2"]) for pair in res]
    print(len(sent1s_concept))
    print(len(sent2s_concept))
    data[7] = sent1s_concept
    data[8] = sent2s_concept
    print(res)
    data.to_csv('./train_concepts.tsv', sep='\t', index=False, header=False)

# "sent": "Watch television do children require to grow up healthy.", "ans": "watch television",
if __name__ == "__main__":
    mua()
