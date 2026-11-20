#!/usr/bin/env python3
"""Configurable ROCOv2 paired feature extraction; train-only text fitting."""
import argparse,json,random
from pathlib import Path
import numpy as np,pandas as pd,torch,joblib
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer,HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from torchvision.models import resnet18,ResNet18_Weights
from torch.utils.data import Dataset,DataLoader

class Images(Dataset):
    def __init__(self,paths,transform):self.paths,self.transform=paths,transform
    def __len__(self):return len(self.paths)
    def __getitem__(self,i):
        with Image.open(self.paths[i]) as im:return self.transform(im.convert('RGB'))

def locate(root,split,ids):
    base=root/split/split
    if not base.is_dir():raise FileNotFoundError(base)
    paths=[base/(str(i)+'.jpg') for i in ids]
    missing=[str(p) for p in paths if not p.is_file()]
    if missing:raise FileNotFoundError(f'{len(missing)} images missing; examples: {missing[:3]}')
    return paths

def color_features(paths):
    # 3 channels * 32 bins = 96 features. Intentionally a nonsemantic baseline.
    rows=[]
    for path in paths:
        with Image.open(path) as im:
            pix=np.asarray(im.convert('RGB').resize((128,128)),dtype=np.uint8)
        rows.append(np.concatenate([np.histogram(pix[:,:,c],bins=32,range=(0,256),density=True)[0] for c in range(3)]))
    return np.asarray(rows,dtype=np.float32)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',default='data/rocov2');p.add_argument('--output',required=True)
    p.add_argument('--train-size',type=int,default=8000);p.add_argument('--validation-size',type=int,default=1000);p.add_argument('--eval-size',type=int,default=1000)
    p.add_argument('--batch-size',type=int,default=32);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    p.add_argument('--image-encoder',choices=['resnet18','color-histogram'],default='resnet18')
    p.add_argument('--text-vectorizer',choices=['tfidf','hashing'],default='tfidf')
    p.add_argument('--text-max-features',type=int,default=12000)
    p.add_argument('--text-components',type=int,default=128)
    p.add_argument('--text-ngram-max',type=int,choices=[1,2],default=2)
    a=p.parse_args();root=Path(a.source);out=Path(a.output)
    if a.train_size<4 or a.validation_size<2 or a.eval_size<2 or a.batch_size<1 or a.text_max_features<2 or a.text_components<2:p.error('Invalid size parameters')
    if out.exists() and any(out.iterdir()):p.error('Output directory must be empty')
    frames={}
    for split,n in (('train',a.train_size+a.validation_size),('valid',a.eval_size)):
        df=pd.read_csv(root/f'{split}_captions.csv',dtype={'ID':str})
        if not {'ID','Caption'}.issubset(df):raise ValueError('Missing ID/Caption')
        df=df.dropna(subset=['ID','Caption']).drop_duplicates('ID')
        if len(df)<n:raise ValueError(f'{split}: only {len(df)} rows; requested {n}')
        df=df.sample(n=n,random_state=a.seed).reset_index(drop=True)
        df['image_path']=[str(x) for x in locate(root,split,df.ID)]
        frames[split]=df
    if a.image_encoder=='resnet18':
        print('Loading pretrained ResNet18...',flush=True)
        weights=ResNet18_Weights.DEFAULT;model=resnet18(weights=weights);model.fc=torch.nn.Identity()
        device=('mps' if torch.backends.mps.is_available() else 'cpu') if a.device=='auto' else a.device
        if device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
        model=model.to(device).eval()
        def encode(paths):
            loader=DataLoader(Images(paths,weights.transforms()),batch_size=a.batch_size,shuffle=False,num_workers=0)
            with torch.inference_mode():return np.concatenate([model(batch.to(device)).cpu().numpy() for batch in loader])
    else:
        device='cpu';encode=color_features
    img={s:encode(df.image_path.tolist()) for s,df in frames.items()}
    def fit_text(training_captions, other_captions):
        if a.text_vectorizer=='tfidf':
            vectorizer=TfidfVectorizer(stop_words='english',max_features=a.text_max_features,
                min_df=2,ngram_range=(1,a.text_ngram_max),dtype=np.float32)
            train_t=vectorizer.fit_transform(training_captions)
        else:
            vectorizer=HashingVectorizer(n_features=a.text_max_features,alternate_sign=False,
                norm='l2',stop_words='english',ngram_range=(1,a.text_ngram_max),dtype=np.float32)
            train_t=vectorizer.transform(training_captions)
        other_t=vectorizer.transform(other_captions)
        k=min(a.text_components,train_t.shape[0]-1,train_t.shape[1]-1)
        if k<2:raise ValueError('Too few text features for SVD')
        svd=TruncatedSVD(n_components=k,random_state=a.seed)
        return vectorizer,svd,svd.fit_transform(train_t),svd.transform(other_t),k

    # Split the official training sample BEFORE fitting any text transformations.
    # The official validation sample is reserved exclusively for final testing.
    train_df=frames['train'].iloc[:a.train_size].copy()
    val_df=frames['train'].iloc[a.train_size:].copy()
    test_df=frames['valid']
    train_img=img['train'][:a.train_size]
    val_img=img['train'][a.train_size:]
    test_img=img['valid']
    vectorizer,svd,train_txt,other_txt,k=fit_text(
        train_df.Caption.astype(str),
        pd.concat([val_df.Caption.astype(str),test_df.Caption.astype(str)],ignore_index=True))
    val_txt=other_txt[:len(val_df)]
    test_txt=other_txt[len(val_df):]
    if not all(np.isfinite(v).all() for v in (train_img,val_img,test_img,train_txt,val_txt,test_txt)):
        raise FloatingPointError('Nonfinite features')
    out.mkdir(parents=True,exist_ok=True)
    names=[f'mod1__{i}' for i in range(train_img.shape[1])]+[f'mod2__{i}' for i in range(k)]
    for name,df,im,txt in (('train',train_df,train_img,train_txt),
                            ('validation',val_df,val_img,val_txt),
                            ('eval',test_df,test_img,test_txt)):
        tab=pd.DataFrame(np.concatenate([im,txt],axis=1),columns=names)
        tab.insert(0,'sample_id',df.ID.to_numpy())
        tab.to_csv(out/f'{name}.csv',index=False)
        df[['ID','Caption','image_path']].to_csv(out/f'{name}_manifest.csv',index=False)
    joblib.dump({'vectorizer':vectorizer,'tfidf':vectorizer if a.text_vectorizer=='tfidf' else None,
                 'svd':svd,'settings':vars(a)},out/'text_feature_pipeline.joblib')
    info={'dataset':'ROCOv2','train_rows':len(train_df),'validation_rows':len(val_df),
          'eval_rows':len(test_df),'image_features':train_img.shape[1],'text_features':k,
          'image_encoder':a.image_encoder,'text_vectorizer':a.text_vectorizer,
          'text_ngram_max':a.text_ngram_max,'text_max_features':a.text_max_features,
          'feature_extraction_device':device,'seed':a.seed,
          'evaluation_split':'official validation subset (held-out test)',
          'text_fitting':'training split only; validation and test transformed without fitting',
          'validation_isolation':'separate train-only fitted validation.csv',
          'limitations':['Not end-to-end CLIP training','ImageNet ResNet18 is not radiology-specific',
                         'Color histogram is a nonsemantic image baseline']}
    (out/'dataset_metadata.json').write_text(json.dumps(info,indent=2));print(json.dumps(info,indent=2))
if __name__=='__main__':main()
