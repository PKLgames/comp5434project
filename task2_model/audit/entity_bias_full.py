import pyarrow as pa, pyarrow.csv as pc, numpy as np, time, json
t0=time.time()
def log(*a): print("[%5.0fs]"%(time.time()-t0), *a, flush=True)
P="/home/pklgames/coding/comp5434project/Dataset2_metadata/train.csv"
tab=pc.read_csv(P, read_options=pc.ReadOptions(block_size=1<<26, use_threads=True),
                convert_options=pc.ConvertOptions(include_columns=["id","rating","user_id","prod_id","parent_prod_id"]))
log("loaded", tab.num_rows, "%.0f MB"%(tab.nbytes/1e6))
ids=tab["id"].to_numpy(zero_copy_only=False)
y=tab["rating"].to_numpy(zero_copy_only=False).astype(np.float64)
def codes(name):
    c=tab[name].combine_chunks(); d=c.dictionary_encode()
    log("dict",name,len(d.dictionary)); return d.indices.to_numpy().astype(np.int32), len(d.dictionary)
uc,nu=codes("user_id"); ic,ni=codes("prod_id"); ac,na=codes("parent_prod_id")
tab=None
val=(ids%40)==0; tr=~val
log("train",tr.sum(),"valid",val.sum())
def st(c):
    MM=int(c.max())+1
    s=np.bincount(c[tr],weights=y[tr],minlength=MM); n=np.bincount(c[tr],minlength=MM).astype(np.float64)
    return s,n
us,un=st(uc); is_,in_=st(ic); as_,an=st(ac)
log("user n median %.0f mean %.2f; item n median %.0f mean %.2f"%(
    np.median(un),un.mean(),np.median(in_),in_.mean()))
log("valid unseen: users %.3f items %.3f parents %.3f"%((un[uc[val]]==0).mean(),(in_[ic[val]]==0).mean(),(an[ac[val]]==0).mean()))
mu=float(y[tr].mean())
K=5.0
def dev(c,s,n):
    out=np.zeros(c.size,dtype=np.float32)
    ci=c[tr]; v=y[tr]
    out[tr]=((s[ci]-v+K*mu)/((n[ci]-1)+K)-mu).astype(np.float32)
    c2=c[~tr]; out[~tr]=((s[c2]+K*mu)/(n[c2]+K)-mu).astype(np.float32)
    return out
def cnt(c,s,n):
    out=np.zeros(c.size,dtype=np.float32); ci=c[tr]; out[tr]=np.log1p(np.maximum(n[ci]-1,0)).astype(np.float32)
    c2=c[~tr]; out[~tr]=np.log1p(n[c2]).astype(np.float32); return out
FU=dev(uc,us,un); FI=dev(ic,is_,in_); FA=dev(ac,as_,an)
VU=cnt(uc,us,un); VI=cnt(ic,is_,in_); VA=cnt(ac,as_,an)
def rmse(p,yy): return float(np.sqrt(np.mean((np.clip(p,1,5)-yy)**2)))
res={"constant":rmse(np.full(int(val.sum()),mu),y[val]),"mu":mu,"n_train":int(tr.sum()),"n_valid":int(val.sum()),
     "entities":{"users":nu,"items":ni,"parents":na}}
log("constant %.4f"%res["constant"])
def fit(cols,kn=K):
    Mtr=np.column_stack([np.ones(int(tr.sum()))]+[c[tr] for c in cols])
    w=np.linalg.lstsq(Mtr,y[tr],rcond=None)[0]
    Mva=np.column_stack([np.ones(int(val.sum()))]+[c[val] for c in cols])
    return rmse(Mva@w,y[val])
for nm,cols in [("item_mean",[FI]),("parent_mean",[FA]),("user_mean",[FU]),
                ("item+parent",[FI,FA]),("user+item",[FU,FI]),
                ("user+item+parent",[FU,FI,FA]),("full",[FU,FI,FA,VU,VI,VA])]:
    res[nm]=fit(cols); log("%-18s %.4f"%(nm,res[nm]))
np.savez("/tmp/v6_ids.npz", uc=uc, ic=ic, ac=ac, y=y.astype(np.int8), val=val,
         us=us,un=un, is_=is_, in_=in_, as_=as_, an=an, mu=mu)
json.dump(res,open("/tmp/v6a.json","w"),indent=1)
log("DONE",json.dumps({k:(round(v,4) if isinstance(v,float) else v) for k,v in res.items()}))
