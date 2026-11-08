import pyarrow.csv as pc, json, time, collections, numpy as np
t0=time.time()
def log(*a): print("[%5.0fs]"%(time.time()-t0), *a, flush=True)
LO,HI=3_000_000,3_020_000
rdr=pc.open_csv("/home/pklgames/coding/comp5434project/Dataset2_metadata/train.csv",
    read_options=pc.ReadOptions(block_size=1<<25),
    convert_options=pc.ConvertOptions(include_columns=["id","rating","comment","time","votes"]))
samp={}; pos=0
for b in rdr:
    ids=b["id"].to_numpy(zero_copy_only=False)
    if ids.max()<LO or ids.min()>HI: pos+=len(ids); continue
    cm=b["comment"].to_pylist(); ra=b["rating"].to_numpy(zero_copy_only=False)
    tm=b["time"].to_numpy(zero_copy_only=False); vo=b["votes"].to_numpy(zero_copy_only=False)
    for k in range(len(ids)):
        if LO<=ids[k]<HI: samp[cm[k]]=(int(ids[k]),int(ra[k]),int(tm[k]),int(vo[k]))
    pos+=len(ids)
    if ids.min()>HI: break
log("our rows sampled",len(samp),"from ids",LO,HI)
rows=[]
with open("/tmp/grocery.jsonl","r",encoding="utf-8") as f:
    n=0
    for line in f:
        n+=1
        if '"text": "' not in line: continue
        try: d=json.loads(line)
        except Exception: continue
        t=d.get("text")
        if t in samp:
            o=samp[t]; rows.append((o[0],o[1],int(d.get("rating") or 0),int(d.get("timestamp") or 0)==o[2],
                                    int(d.get("helpful_vote") or 0)==o[3],o[2]))
        if n%10000000==0: log("raw",n,"hits",len(rows))
log("hits",len(rows),"distinct our ids",len({r[0] for r in rows}))
seen=set(); uq=[]
for r in rows:
    if r[0] in seen: continue
    seen.add(r[0]); uq.append(r)
ts=[r for r in uq if r[3]]
log("matched(ts identical) %d ; rating identical %d (%.2f%%)"%(len(ts),sum(1 for r in ts if r[1]==r[2]),
    100*sum(1 for r in ts if r[1]==r[2])/max(len(ts),1)))
cm=collections.Counter((r[1],r[2]) for r in ts)
print("confusion our(row)->raw(col):")
for a in range(1,6): print("  our",a,":",[cm.get((a,b),0) for b in range(1,6)])
ys=np.array([r[1] for r in ts],float); xs=np.array([r[2] for r in ts],float)
print("mismatch rate per raw value:", {int(v):round(float(np.mean(ys[xs==v]!=v)),3) for v in range(1,6)})
print("our marginal:",np.round(np.bincount(ys.astype(int),minlength=6)[1:]/len(ys),3),
      "raw marginal:",np.round(np.bincount(xs.astype(int),minlength=6)[1:]/len(xs),3))
# Var(y|raw) = MSE floor if we knew raw perfectly
floor=[]
for v in range(1,6):
    m=ys[xs==v]
    if len(m): floor.append(len(m)*(m.var()))
print("E[Var(our|raw)] = %.4f -> RMSE floor given perfect raw label = %.4f"%(sum(floor)/len(ys),np.sqrt(sum(floor)/len(ys))))
ids=np.array([r[0] for r in ts]); mism=(ys!=xs).astype(int)
for k in (2,3,4,5,7,10):
    rates={int(v):round(float(mism[ids%k==v].mean()),3) for v in range(k)}
    print("mismatch by id%%%d:"%k, rates, "counts", {int(v):int((ids%k==v).sum()) for v in range(k)})
yr=(np.array([r[5] for r in ts])//1000).astype("datetime64[s]").astype("datetime64[Y]").astype(int)+1970
print("mismatch by year:", {int(y):round(float(mism[yr==y].mean()),3) for y in np.unique(yr)})
print("replacement distribution among mismatches:", np.round(np.bincount(ys[mism==1].astype(int),minlength=6)[1:]/mism.sum(),3))
json.dump({"n":int(len(ts)),"rating_identical":int(sum(1 for r in ts if r[1]==r[2])),
           "floor_var":float(sum(floor)/len(ys))},open("/tmp/v8.json","w"),indent=1)
log("DONE")
