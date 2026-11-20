#!/usr/bin/env python3
"""Configurable, train-fitted sparse 10x RNA/ATAC preprocessing tool for agent selection."""
import argparse, json, warnings
from pathlib import Path
import numpy as np, pandas as pd, h5py, joblib
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',default='data/multiome/filtered_feature_bc_matrix.h5')
    p.add_argument('--output',required=True)
    p.add_argument('--train-size',type=int,default=2500)
    p.add_argument('--eval-size',type=int,default=500)
    p.add_argument('--validation-size',type=int,default=0,help='Independent validation cells; fitted transforms use train cells only')
    p.add_argument('--rna-features',type=int,default=2000)
    p.add_argument('--atac-features',type=int,default=12000)
    p.add_argument('--rna-components',type=int,default=64)
    p.add_argument('--atac-components',type=int,default=64)
    p.add_argument('--rna-normalization',choices=['library-log1p','log1p'],default='library-log1p')
    p.add_argument('--rna-feature-selection',choices=['variance','detection'],default='variance')
    p.add_argument('--atac-feature-selection',choices=['frequency','variance'],default='frequency')
    p.add_argument('--atac-transform',choices=['tfidf','binary-tfidf'],default='tfidf')
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();out=Path(a.output)
    # Avoid problematic multi-threaded BLAS reductions during sparse SVD.
    threadpool_limits(limits=1)
    if out.exists() and any(out.iterdir()):p.error('Output directory must be empty')
    source=Path(a.source)
    if source.is_dir():source=source/'filtered_feature_bc_matrix.h5'
    with h5py.File(source) as h:
        m=h['matrix'];shape=tuple(int(x) for x in m['shape'][:]);barcodes=np.array([v.decode() for v in m['barcodes'][:]])
        types=np.array([v.decode() for v in m['features']['feature_type'][:]])
        # 10x CSC matrix is feature-by-cell. Transpose to cell-by-feature CSR.
        x=sparse.csc_matrix((m['data'][:],m['indices'][:],m['indptr'][:]),shape=shape).T.tocsr()
    rna=x[:,np.flatnonzero(types=='Gene Expression')].tocsr()
    atac=x[:,np.flatnonzero(types=='Peaks')].tocsr()
    del x
    if len(barcodes)<a.train_size+a.validation_size+a.eval_size:raise ValueError('Requested more cells than available')
    rng=np.random.default_rng(a.seed);chosen=rng.permutation(len(barcodes))[:a.train_size+a.validation_size+a.eval_size]
    tr=chosen[:a.train_size];va=chosen[a.train_size:a.train_size+a.validation_size];ev=chosen[a.train_size+a.validation_size:];splits={'train':tr,'eval':ev}
    if a.validation_size: splits['validation']=va
    # RNA feature ranking uses TRAIN CELLS ONLY; library normalization per cell.
    rtr=rna[tr].astype(np.float64)
    if a.rna_normalization=='library-log1p':
        sizes=np.asarray(rtr.sum(axis=1)).ravel();rtr=sparse.diags(1e4/np.maximum(sizes,1))@rtr
    rtr=rtr.tocsr();rtr.data=np.log1p(rtr.data)
    mean=np.asarray(rtr.mean(axis=0)).ravel()
    sqmean=np.asarray(rtr.power(2).mean(axis=0)).ravel()
    variance=np.maximum(sqmean-mean**2,0)
    rna_score=variance if a.rna_feature_selection=='variance' else np.asarray((rtr>0).sum(axis=0)).ravel()
    ri=np.argsort(rna_score)[-min(a.rna_features,rna.shape[1]):]
    if len(ri)<=a.rna_components:raise ValueError('rna-components must be smaller than selected features')
    def transform_rna(indices):
        z=rna[indices][:,ri].astype(np.float64)
        if a.rna_normalization=='library-log1p':
            # Library size must be computed over ALL genes, not selected genes.
            sizes=np.asarray(rna[indices].sum(axis=1)).ravel()
            z=sparse.diags(1e4/np.maximum(sizes,1))@z
        z=z.tocsr();z.data=np.log1p(z.data);return z
    # ATAC peak selection uses training-cell document frequency only.
    atr=atac[tr].astype(np.float64).tocsr();df=np.asarray((atr>0).sum(axis=0)).ravel()
    atac_score=df
    if a.atac_feature_selection=='variance':
        amean=np.asarray(atr.mean(axis=0)).ravel()
        asq=np.asarray(atr.power(2).mean(axis=0)).ravel()
        atac_score=np.maximum(asq-amean**2,0)
    ai=np.argsort(atac_score)[-min(a.atac_features,atac.shape[1]):]
    if len(ai)<=a.atac_components:raise ValueError('atac-components must be smaller than selected features')
    def transform_atac(indices):
        z=atac[indices][:,ai].astype(np.float64)
        if a.atac_transform=='binary-tfidf':z.data[:]=1
        return z
    atr=transform_atac(tr)
    idf=np.log1p((len(tr)+1)/(np.asarray((atr>0).sum(axis=0)).ravel()+1)).astype(np.float64)
    def tfidf(z):return normalize(z,norm='l1',axis=1).multiply(idf).tocsr()
    # Capture numerical warnings from randomized SVD. Finite outputs alone do
    # not prove the decomposition was numerically reliable.
    svd_diagnostics = {}
    def fit_checked_svd(name, matrix, n_components):
        if not np.isfinite(matrix.data).all():
            raise FloatingPointError(f"Nonfinite {name} input to SVD")
        if n_components >= min(matrix.shape):
            raise ValueError(f"{name}: components must be less than min(matrix.shape)")
        attempts = []
        for algorithm in ('randomized', 'arpack'):
            model = TruncatedSVD(n_components=n_components, algorithm=algorithm,
                                 random_state=a.seed)
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always', RuntimeWarning)
                    model.fit(matrix)
                numerical_warnings = [str(w.message) for w in caught
                                      if issubclass(w.category, RuntimeWarning)]
                finite = all(np.isfinite(v).all() for v in
                             (model.components_, model.singular_values_,
                              model.explained_variance_, model.explained_variance_ratio_))
                attempts.append({'algorithm': algorithm,
                                 'runtime_warnings': numerical_warnings,
                                 'finite_outputs': bool(finite)})
                if numerical_warnings or not finite:
                    continue
                svd_diagnostics[name] = {
                    'selected_algorithm': algorithm,
                    'attempts': attempts,
                    'explained_variance_ratio_sum': float(
                        np.sum(model.explained_variance_ratio_)),
                }
                return model
            except Exception as exc:
                attempts.append({'algorithm': algorithm,
                                 'error': f'{type(exc).__name__}: {exc}'})
        raise RuntimeError(f"{name} SVD failed numerical checks: {attempts}")

    rsvd = fit_checked_svd('RNA', transform_rna(tr), a.rna_components)
    asvd = fit_checked_svd('ATAC', tfidf(atr), a.atac_components)
    out.mkdir(parents=True,exist_ok=True)
    for split,indices in splits.items():
        r=rsvd.transform(transform_rna(indices));t=asvd.transform(tfidf(transform_atac(indices)))
        if not np.isfinite(r).all() or not np.isfinite(t).all():
            raise FloatingPointError(f"Nonfinite reduced features in {split}")
        tab=pd.DataFrame(np.concatenate([r,t],axis=1),columns=[f'mod1__{j}' for j in range(r.shape[1])]+[f'mod2__{j}' for j in range(t.shape[1])]);tab.insert(0,'sample_id',barcodes[indices]);tab.to_csv(out/f'{split}.csv',index=False)
        pd.DataFrame({'sample_id':barcodes[indices]}).to_csv(out/f'{split}_manifest.csv',index=False)
    info={'dataset':'10x Genomics 10k PBMC Multiome','source':str(source),'train_rows':len(tr),'eval_rows':len(ev),'validation_rows':len(va),'paired_by':'cell barcode','has_cell_type_labels':False,'rna_features_selected':len(ri),'atac_peaks_selected':len(ai),'rna_components':a.rna_components,'atac_components':a.atac_components,'rna_normalization':a.rna_normalization,'rna_feature_selection':a.rna_feature_selection,'atac_feature_selection':a.atac_feature_selection,'atac_transform':a.atac_transform,'selection_and_fitting':'train only','svd_diagnostics':svd_diagnostics,'seed':a.seed,'limitations':['No ground-truth cell-type labels in this file','RNA/ATAC SVD features are fixed before neural representation learning']}
    joblib.dump({'rna_feature_indices':ri,'atac_feature_indices':ai,'atac_idf':idf,'rna_svd':rsvd,'atac_svd':asvd,'settings':vars(a)},out/'feature_pipeline.joblib')
    (out/'dataset_metadata.json').write_text(json.dumps(info,indent=2));(out/'svd_diagnostics.json').write_text(json.dumps(svd_diagnostics,indent=2));print(json.dumps(info,indent=2))
if __name__=='__main__':main()
