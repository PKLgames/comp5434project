"""Noise-aware redesign experiment: clean-label target vs delivered-label target."""
import pyarrow.csv as pc, numpy as np, time, json
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import Ridge
from scipy.sparse import hstack as sph, csr_matrix
t0=time.time()
def log(*a): print("[%5.0fs]"%(time.time()-t0), *a, flush=True)
CSV="/home/pklgames/coding/comp5434project/Dataset2_metadata/train.csv"
N=7684645
d=np.load("/tmp/clean_y.npz"); cy=d["cy"].astype(np.float64); matched=d["matched"]; ag=d["ag"].astype(np.float64)
log("clean labels loaded; coverage %.3f"%(matched.mean()))
# entity codes
tab=pc.read_csv(CSV, convert_options=pc.ConvertOptions(include_columns=["id","user_id","prod_id"]))
ids=tab["id"].to_numpy(zero_copy_only=False)
def enc(name):
    a=tab[name].combine_chunks().dictionary_encode(); return a.indices.to_numpy().astype(np.int32), len(a.dictionary)
uc,nu=enc("user_id"); ic,ni=enc("prod_id"); del tab
log("codes users",nu,"items",ni)
val=(ids%40)==0
pool=((ids%8)==0)&(~val)
log("valid",int(val.sum()),"train pool",int(pool.sum()))
ytrain_clean=np.where(matched,cy,ag)
K=5.0; mu=float(ytrain_clean[~val].mean())
def stats(c):
    mm=int(c.max())+1
    s=np.bincount(c[~val],weights=ytrain_clean[~val],minlength=mm)
    n=np.bincount(c[~val],minlength=mm).astype(np.float64)
    return s,n
def feats(c):
    s,n=stats(c)
    dev=np.empty(c.size,np.float32); cnt=np.empty(c.size,np.float32)
    dev[~val]=((s[c[~val]]-ytrain_clean[~val]+K*mu)/((n[c[~val]]-1)+K)-mu).astype(np.float32)
    dev[val]=((s[c[val]]+K*mu)/(n[c[val]]+K)-mu).astype(np.float32)
    cnt[~val]=np.log1p(np.maximum(n[c[~val]]-1,0)).astype(np.float32)
    cnt[val]=np.log1p(n[c[val]]).astype(np.float32)
    return dev,cnt
FU,VU=feats(uc); FI,VI=feats(ic)
log("entity feats done")
# text + numerics for pool+valid rows
sel=pool|val
rdr=pc.open_csv(CSV, read_options=pc.ReadOptions(block_size=1<<25),
    convert_options=pc.ConvertOptions(include_columns=["id","comment","title","votes","purchased","time"]))
docs=[]; rows=[]; num=[]
for b in rdr:
    bi=b["id"].to_numpy(zero_copy_only=False)
    m=sel[bi]
    if not m.any(): continue
    b=b.filter(m); ii=b["id"].to_numpy(zero_copy_only=False)
    cm=b["comment"].to_pylist(); ti=b["title"].to_pylist()
    vo=b["votes"].to_numpy(zero_copy_only=False); pu=b["purchased"].to_numpy(zero_copy_only=False)
    tm=b["time"].to_numpy(zero_copy_only=False).astype(np.int64)
    yr=((tm//1000).astype("datetime64[s]").astype("datetime64[Y]").astype(int)+1970).astype(np.float32)
    for k in range(len(ii)):
        c=cm[k] or ""; t=ti[k] or ""
        docs.append((t+" "+c).lower()); rows.append(int(ii[k]))
        num.append((np.float32(np.log1p(vo[k])), np.float32(pu[k]), yr[k],
                    np.float32(np.log1p(len(c))), np.float32(np.log1p(len(t)))))
log("text rows",len(rows))
rows=np.array(rows); num=np.array(num,dtype=np.float32)
X=HashingVectorizer(n_features=2**17, ngram_range=(1,2), alternate_sign=False, norm=None, dtype=np.float32).transform(docs)
X=TfidfTransformer(sublinear_tf=True).fit_transform(X)
del docs
log("tfidf",X.shape,"nnz/row %.1f"%(X.nnz/X.shape[0]))
X=X.tocsr()
A=sph([X, csr_matrix(np.column_stack([FU[rows],FI[rows],VU[rows],VI[rows],num]))]).tocsr()
del X
log("design",A.shape)
pm=pool[rows]; vm=val[rows]; mm=matched[rows]
i_tr=np.nonzero(pm&mm)[0]; i_va=np.nonzero(vm&mm)[0]
y_clean=cy[rows]; y_ag=ag[rows]
def rmse(p,y): return float(np.sqrt(np.mean((np.clip(p,1,5)-y)**2)))
res={}
for tag,yy,alphas in [("clean",y_clean,(0.3,1.0,3.0)),("delivered",y_ag,(1.0,))]:
    for a in alphas:
        m=Ridge(alpha=a, solver="lsqr", tol=1e-3, max_iter=400)
        m.fit(A[i_tr], yy[i_tr]); p=m.predict(A[i_va])
        res[f"{tag}_a{a}"]={"rmse_clean":rmse(p,y_clean[i_va]),"rmse_delivered":rmse(p,y_ag[i_va]),
                             "rmse_delivered_shrunk":rmse(0.75*p+0.75,y_ag[i_va])}
        log(tag,"alpha",a,res[f"{tag}_a{a}"])
        if tag=="clean" and a==1.0: pass
# affine calibration of clean predictions, fitted on train-pool rows
m=Ridge(alpha=3.0, solver="lsqr", tol=1e-3, max_iter=400).fit(A[i_tr], y_clean[i_tr])
ptr=m.predict(A[i_tr]); ab=np.polyfit(ptr, y_ag[i_tr], 1)
pv=m.predict(A[i_va]); pcal=ab[0]*pv+ab[1]
res["clean_affine"]={"ab":[float(ab[0]),float(ab[1])],"rmse_clean":rmse(pv,y_clean[i_va]),"rmse_delivered":rmse(pcal,y_ag[i_va])}
log("clean_affine",res["clean_affine"])
json.dump(res,open("/tmp/ex2.json","w"),indent=1)
log("DONE")
