#!/usr/bin/env python3
"""Read-only, sampled source inspection for agent preprocessing decisions."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import h5py


def summarize(v):
    v=np.asarray(v,dtype=float)
    return {'median':float(np.median(v)),'p10':float(np.percentile(v,10)),'p90':float(np.percentile(v,90)),'max':float(v.max())} if len(v) else None


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True)
    p.add_argument('--kind',required=True,choices=['multiome','rocov2'])
    p.add_argument('--output')
    p.add_argument('--sample-size',type=int,default=1000)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    if a.sample_size<1:p.error('--sample-size must be positive')
    root=Path(a.source);rng=np.random.default_rng(a.seed)
    if a.kind=='rocov2':
        result={'dataset':'ROCOv2','source':str(root),'paired_by':'image ID and caption','inspection_sample_size_limit':a.sample_size}
        for split in ('train','valid'):
            d=pd.read_csv(root/f'{split}_captions.csv',dtype={'ID':str})
            if not {'ID','Caption'}.issubset(d):raise ValueError('Expected ID and Caption columns')
            valid=d.dropna(subset=['ID','Caption']).drop_duplicates('ID')
            sampled=valid.sample(n=min(a.sample_size,len(valid)),random_state=a.seed)
            captions=sampled.Caption.astype(str)
            words=captions.str.findall(r'\b\w+\b').str.len().to_numpy()
            result[split]={'caption_rows':len(d),'complete_pairs':len(valid),'unique_ids':int(d.ID.nunique()),'sampled_caption_words':summarize(words),'sampled_empty_captions':int((captions.str.strip()=='').sum()),'sampled_unique_caption_fraction':float(captions.nunique()/len(captions)) if len(captions) else None}
        result['supported_preprocessing']={'image_encoder':['resnet18','color-histogram'],'text_vectorizer':['tfidf','hashing'],'text_components':'configurable positive integer (SVD)','text_max_features':'configurable for TF-IDF','text_ngram_max':[1,2]}
        result['decision_guidance']=['Use ResNet18 for semantic image features when weights are cached or available; color-histogram is a low-cost nonsemantic baseline.','Use TF-IDF for interpretable train-fitted vocabulary; hashing for fixed-memory vocabulary-free features.','Choose SVD dimension and n-grams based on caption lengths, sample size and compute budget.']
    else:
        f=root if root.is_file() else root/'filtered_feature_bc_matrix.h5'
        with h5py.File(f) as h:
            m=h['matrix'];shape=tuple(int(x) for x in m['shape'][:]);types=np.asarray([v.decode() for v in m['features']['feature_type'][:]])
            n_cells=shape[1];selected=rng.choice(n_cells,size=min(a.sample_size,n_cells),replace=False)
            # CSC column slicing: read only selected cell segments, not the full matrix.
            indptr=m['indptr'][:];data=m['data'];indices=m['indices']
            rna_idx=np.flatnonzero(types=='Gene Expression');atac_idx=np.flatnonzero(types=='Peaks')
            is_rna=np.zeros(shape[0],dtype=bool);is_rna[rna_idx]=True
            is_atac=np.zeros(shape[0],dtype=bool);is_atac[atac_idx]=True
            rna_counts=[];rna_detected=[];atac_counts=[];atac_detected=[]
            for c in selected:
                lo,hi=int(indptr[c]),int(indptr[c+1]);ii=indices[lo:hi];vv=data[lo:hi]
                rm=is_rna[ii];am=is_atac[ii]
                rna_counts.append(float(vv[rm].sum()));rna_detected.append(int(np.count_nonzero(vv[rm])))
                atac_counts.append(float(vv[am].sum()));atac_detected.append(int(np.count_nonzero(vv[am])))
        result={'dataset':'10x Multiome PBMC','source':str(f),'cells':n_cells,'features':{'Gene Expression':len(rna_idx),'Peaks':len(atac_idx)},'sparse_nonzeros':int(indptr[-1]),'paired_by':'shared cell barcodes','has_cell_type_labels':False,'sampled_cells':len(selected),'rna_library_size':summarize(rna_counts),'rna_detected_genes':summarize(rna_detected),'atac_total_counts':summarize(atac_counts),'atac_detected_peaks':summarize(atac_detected),'supported_preprocessing':{'rna_normalization':['library-log1p','log1p'],'rna_feature_selection':['variance','detection'],'atac_transform':['tfidf','binary-tfidf'],'atac_feature_selection':['frequency','variance'],'rna_features':'configurable','atac_features':'configurable','rna_components':'configurable SVD dimension','atac_components':'configurable SVD dimension'},'decision_guidance':['Prefer library-size normalization if RNA sequencing depth varies substantially.','Binary ATAC TF-IDF discards count magnitudes; useful for accessibility presence.','Fit feature selection, IDF and SVD on training cells only; do not invent cell-type labels.','Never densify the full feature-by-cell matrix.']}
    output=json.dumps(result,indent=2);print(output)
    if a.output:
        dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(output)

if __name__=='__main__':main()
