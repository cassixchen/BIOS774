import argparse, json
from pathlib import Path
import pandas as pd

def inspect(data_dir):
    train=pd.read_csv(Path(data_dir)/'train.csv'); ev=pd.read_csv(Path(data_dir)/'eval.csv')
    mods={m:[c for c in train if c.startswith(m+'__')] for m in ('mod1','mod2')}
    if not all(mods.values()): raise ValueError('Need both mod1__* and mod2__* numeric columns')
    if len(train)<4 or len(ev)<2: raise ValueError('Insufficient training/evaluation rows')
    for m,cols in mods.items():
        if any(c not in ev for c in cols): raise ValueError(f'Missing evaluation columns for {m}')
        for df in (train,ev):
            if any(not pd.api.types.is_numeric_dtype(df[c]) for c in cols): raise ValueError(f'{m} requires numeric features')
    return {'train_rows':len(train),'eval_rows':len(ev),'modalities':{m:{'features':len(cols),'missing_fraction':float(train[cols].isna().mean().mean())} for m,cols in mods.items()},'has_labels':'label' in train and 'label' in ev,'label_classes':int(train.label.nunique()) if 'label' in train else None,'schema_valid':True}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--output',required=True);a=p.parse_args();d=inspect(a.data_dir);Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(d,indent=2));print(json.dumps(d,indent=2))
