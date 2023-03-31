import json
import pandas as pd
import numpy as np
from typing import List
from nlgeval.pycocoevalcap.bleu.bleu import Bleu
from nlgeval.pycocoevalcap.cider.cider import Cider
from nlgeval.pycocoevalcap.meteor.meteor import Meteor
from nlgeval.pycocoevalcap.rouge.rouge import Rouge

def get_generation_eval_scores(raw_hyp_list, raw_ref_list, num_hyp, do_test):
    assert len(raw_hyp_list)==len(raw_ref_list)
    
    hyp_lists = []
    for i in range(num_hyp):
        hyp_lists.append([hyp[i] for hyp in raw_hyp_list])
    ref_list = [[ref.strip() for ref in refs.split('|')] for refs in raw_ref_list]

    #calculate other scores
    hyps = {idx: [strippedlines] for (idx, strippedlines) in enumerate(hyp_lists[0])}
    refs = {idx: strippedlines for (idx, strippedlines) in enumerate(ref_list)}
    bleus = [str(100*round(score, 3)) for score in Bleu(4).compute_score(refs, hyps)[0]]
    cider = Cider().compute_score(refs, hyps)[0]
    if do_test:
        meteor = Meteor().compute_score(refs, hyps)[0]
    else:
        meteor = 0.0
    rouge = Rouge().compute_score(refs, hyps)[0]
    metrics = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr"]
    scores =  bleus + [str(100*round(meteor, 3)), str(100*round(rouge, 3)), str(100*round(cider, 3))]
    return metrics, scores