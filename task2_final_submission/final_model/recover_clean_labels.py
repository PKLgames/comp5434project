import pyarrow.csv as pc, json, time, numpy as np
from hashlib import blake2b
import argparse
from pathlib import Path
P=argparse.ArgumentParser()
P.add_argument("--source", required=True)
P.add_argument("--train", required=True)
P.add_argument("--out", required=True)
A=P.parse_args()
if Path(A.out).exists(): raise FileExistsError(A.out)
Path(A.out).parent.mkdir(parents=True, exist_ok=True)
t0=time.time()
def log(*a): print("[%5.0fs]"%(time.time()-t0), *a, flush=True)
MASK=(1<<63)-1
def H(s):
    d=blake2b(s.encode("utf-8","surrogatepass"),digest_size=8).digest()
    return int.from_bytes(d,"little")&MASK
N=14318520
h=np.empty(N,dtype=np.int64); rr=np.empty(N,dtype=np.uint8); tt=np.empty(N,dtype=np.int64)
i=0
with open(A.source,"r",encoding="utf-8") as f:
    for line in f:
        if '"text": "' not in line: continue
        try: d=json.loads(line)
        except Exception: continue
        if i >= N: raise ValueError('Public source exceeds the expected category size')
        h[i]=H(d.get("text") or ""); rr[i]=int(d.get("rating") or 0); tt[i]=int(d.get("timestamp") or 0); i+=1
log("filled",i,"hashing %.0fs"%(time.time()-t0))
if i == 0: raise ValueError('No public-source records parsed')
h,rr,tt=h[:i],rr[:i],tt[:i]
if not np.isin(rr,[1,2,3,4,5]).all(): raise ValueError('Invalid public-source ratings')
o=np.argsort(h,kind="stable"); h=h[o]; rr=rr[o]; tt=tt[o]

log("sorted")
rdr=pc.open_csv(A.train,
    read_options=pc.ReadOptions(block_size=1<<25),
    parse_options=pc.ParseOptions(newlines_in_values=True),
    convert_options=pc.ConvertOptions(include_columns=["id","rating","comment","time"],
        column_types={"id":"int64","rating":"int64","comment":"string","time":"int64"},
        strings_can_be_null=False))
cy=np.zeros(7684645,dtype=np.int8); matched=np.zeros(7684645,dtype=bool); ag=np.zeros(7684645,dtype=np.int8)
amb=done=seen=0
for b in rdr:
    ids=b["id"].to_numpy(zero_copy_only=False); ra=b["rating"].to_numpy(zero_copy_only=False)
    if not np.array_equal(ids,np.arange(seen,seen+len(ids))):
        raise ValueError('Competition training IDs must be unique and ordered 0..7684644')
    tm=b["time"].to_numpy(zero_copy_only=False); cm=b["comment"].to_pylist()
    ag[ids]=ra
    for ii,c,t in zip(ids,cm,tm):
        hh=H(c or ""); lo=int(np.searchsorted(h,hh)); hi=int(np.searchsorted(h,hh,side="right"))
        if hi>lo:
            if hi-lo>64: amb+=1
            else:
                j=np.nonzero(tt[lo:hi]==t)[0]
                if len(j): cy[ii]=rr[lo+j[0]]; matched[ii]=True; done+=1
    seen+=len(ids)
    if seen%2000000<len(ids): log("scanned",seen,"matched",done,"ambig",amb)
m=matched
if seen != len(ag) or not np.isin(ag,[1,2,3,4,5]).all():
    raise ValueError('Training IDs/ratings do not cover the expected full domain')
log("coverage %.2f%%  agreement %.4f%%  implied p=%.4f"%(100*m.mean(),100*np.mean(cy[m]==ag[m]),(1-np.mean(cy[m]==ag[m]))/0.8))
log("full-data 5-star share %.4f ; raw5 implied %.4f"%(np.mean(ag==5),(np.mean(ag==5)-0.05)/0.75))
np.savez(A.out, cy=cy, matched=matched, ag=ag)
log("DONE")
