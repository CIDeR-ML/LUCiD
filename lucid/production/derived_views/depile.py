#!/usr/bin/env python3
"""De-pile a pile-up dataset: split each event's N interactions into N separate
single-interaction events (Option A — one folder, ~2x events).

ROOT-free: pure re-indexing of the stored sensor/hits/step/labl files.
- per_track / per_particle / per_interaction sliced to one interaction (FKs remapped)
- step segments + sensor_hits filtered & remapped; hits filtered & remapped
- sensor re-aggregated from that interaction's sensor_hits
- the per-segment 'edep' VALUE dataset is preserved
Source event e (interactions 0..M-1) -> dest events [e*M_max ... ] sequential.

Processes ONE file index (resumable: re-run skips files whose dest event count
already matches). Usage: depile.py SRC_DIR DST_DIR FILE_INDEX

Dataset-agnostic: works on any pile-up dataset. See submit_depile_array.sbatch
in this directory for the SLURM array driver, and README.md for the derived-view
family (relabel_*.py) and the hard-won gotchas.
"""
import sys, os, glob, numpy as np, h5py

SRC, DST, FIDX = sys.argv[1], sys.argv[2], int(sys.argv[3])
tag = f"{FIDX:04d}"

def remap(n, keep):
    m = np.full(n, -1, np.int64); m[keep] = np.arange(int(keep.sum())); return m
def csr_slice(data, off, keep_idx):
    outd=[]; outoff=[0]
    for r in keep_idx:
        outd.append(data[off[r]:off[r+1]]); outoff.append(outoff[-1]+(off[r+1]-off[r]))
    cat = np.concatenate(outd) if outd else data[:0]
    return cat, np.array(outoff, dtype=off.dtype)

sf=f"{SRC}/sensor/wc_sensor_{tag}.h5"; hf=f"{SRC}/hits/wc_hits_{tag}.h5"
gf=f"{SRC}/step/wc_step_{tag}.h5";     lf=f"{SRC}/labl/wc_labl_{tag}.h5"
for p in (sf,hf,gf,lf):
    if not os.path.exists(p): print(f"missing {p}"); sys.exit(2)

# count source events
with h5py.File(lf,"r") as h:
    src_events=sorted(k for k in h if k.startswith("event_"))

# resumable skip: dest already has the expected number of events?
od=f"{DST}/labl/wc_labl_{tag}.h5"
if os.path.exists(od):
    try:
        with h5py.File(lf,"r") as a, h5py.File(od,"r") as b:
            exp=sum(a[e]["per_interaction"]["t0"].shape[0] for e in src_events)
            got=sum(1 for k in b if k.startswith("event_"))
            if got==exp and exp>0:
                print(f"[{tag}] skip (done: {got} events)"); sys.exit(0)
    except Exception: pass

for d in ("sensor","hits","step","labl"):
    os.makedirs(f"{DST}/{d}", exist_ok=True)

So = h5py.File(sf,"r"); Ho=h5py.File(hf,"r"); Go=h5py.File(gf,"r"); Lo=h5py.File(lf,"r")
sP=f"{DST}/sensor/wc_sensor_{tag}.h5"; hP=f"{DST}/hits/wc_hits_{tag}.h5"
gP=f"{DST}/step/wc_step_{tag}.h5";     lP=f"{DST}/labl/wc_labl_{tag}.h5"
Sw=h5py.File(sP+".part","w"); Hw=h5py.File(hP+".part","w"); Gw=h5py.File(gP+".part","w"); Lw=h5py.File(lP+".part","w")
GZ=dict(compression="gzip", compression_opts=4)

# copy config groups verbatim, fix source_event_idx + n_events later
def copy_config(src, dst):
    src.copy("config", dst)

copy_config(So,Sw); copy_config(Ho,Hw); copy_config(Go,Gw); copy_config(Lo,Lw)

out_src_idx=[]; out=0
for e in src_events:
    Lg=Lo[e]; pi={k:Lg["per_interaction"][k][:] for k in Lg["per_interaction"]}
    nint=pi["t0"].shape[0]
    pt={k:Lg["per_track"][k][:] for k in Lg["per_track"]}
    pp={k:Lg["per_particle"][k][:] for k in Lg["per_particle"]}
    pe_t0=Lg["per_event"]["t0"][()] if "per_event" in Lg and "t0" in Lg["per_event"] else None
    src_idx=int(Lo["config/source_event_idx"][src_events.index(e)]) if "config" in Lo else int(Lg.attrs.get("source_event_idx",out))
    Sg=So[e]; sen={k:Sg[k][:] for k in Sg}
    Hg=Ho[e]; hit={k:Hg[k][:] for k in Hg}
    Gg=Go[e]; seg={k:Gg[k][:] for k in Gg if k!="sensor_hits"}; sh={k:Gg["sensor_hits"][k][:] for k in Gg["sensor_hits"]}

    for K in range(nint):
        kt = pt["interaction"]==K
        kp = pp["interaction_idx"]==K
        tmap=remap(pt["interaction"].shape[0],kt); pmap=remap(pp["interaction_idx"].shape[0],kp)
        kseg = kt[seg["track_idx"]]; smap=remap(seg["track_idx"].shape[0],kseg)
        ksh = kseg[sh["segment_idx"]]
        kh = kp[hit["particle_idx"]]
        en=f"event_{out:03d}"
        # ---- labl ----
        lg=Lw.create_group(en)
        gpt=lg.create_group("per_track")
        for k,v in pt.items():
            nv=v[kt]
            if k=="particle_idx": nv=np.where(nv>=0,pmap[nv],-1).astype(v.dtype)
            elif k=="interaction": nv=np.zeros(int(kt.sum()),v.dtype)
            gpt.create_dataset(k,data=nv)
        gpp=lg.create_group("per_particle")
        for k in ("category","contained","interaction_idx"):
            nv=pp[k][kp];
            if k=="interaction_idx": nv=np.zeros(int(kp.sum()),pp[k].dtype)
            gpp.create_dataset(k,data=nv)
        kpidx=np.where(kp)[0]
        for nm in ("genealogy","ext_genealogy"):
            d,o=csr_slice(pp[nm+"_data"],pp[nm+"_offsets"],kpidx); gpp.create_dataset(nm+"_data",data=d); gpp.create_dataset(nm+"_offsets",data=o)
        gpi=lg.create_group("per_interaction")
        for k,v in pi.items():
            if k.endswith("_data") or k.endswith("_offsets"): continue
            gpi.create_dataset(k,data=v[[K]])
        for base in ("primary_track_ids","primary_pdgs","primary_energies"):
            if base+"_data" in pi:
                d,o=csr_slice(pi[base+"_data"],pi[base+"_offsets"],[K]); gpi.create_dataset(base+"_data",data=d); gpi.create_dataset(base+"_offsets",data=o)
        gpe=lg.create_group("per_event")
        gpe.create_dataset("t0",data=np.array(pi["t0"][K],dtype=pi["t0"].dtype))
        if "contained" in pi: gpe.create_dataset("contained",data=np.array(bool(pi["contained"][K])))
        lg.attrs["source_event_idx"]=np.uint32(src_idx); lg.attrs["n_particles"]=np.int64(int(kp.sum())); lg.attrs["n_tracks"]=np.int64(int(kt.sum()))
        # ---- step ----
        gg=Gw.create_group(en)
        for k,v in seg.items():
            nv=v[kseg]
            if k=="track_idx": nv=tmap[nv].astype(v.dtype)
            gg.create_dataset(k,data=nv,**GZ)
        gsh=gg.create_group("sensor_hits")
        for k,v in sh.items():
            nv=v[ksh]
            if k=="segment_idx": nv=smap[nv].astype(v.dtype)
            gsh.create_dataset(k,data=nv,**GZ)
        gg.attrs["source_event_idx"]=np.uint32(src_idx); gg.attrs["n_tracks"]=np.int64(int(kt.sum())); gg.attrs["n_segments"]=np.int64(int(kseg.sum())); gg.attrs["has_segment_sensor_map"]=np.True_
        gsh.attrs["n_segment_hits"]=np.int64(int(ksh.sum()))
        # ---- hits ----
        hg=Hw.create_group(en)
        for k,v in hit.items():
            nv=v[kh]
            if k=="particle_idx": nv=pmap[nv].astype(v.dtype)
            hg.create_dataset(k,data=nv,**GZ)
        hg.attrs["source_event_idx"]=np.uint32(src_idx); hg.attrs["n_particles"]=np.int64(int(kp.sum())); hg.attrs["n_particle_hits"]=np.int64(int(kh.sum()))
        # ---- sensor: re-aggregate from this interaction's sensor_hits ----
        ksid=sh["sensor_idx"][ksh]; kpe=sh["PE"][ksh]; kt_=sh["T"][ksh]
        if ksid.size:
            order=np.argsort(ksid); ksid=ksid[order]; kpe=kpe[order]; kt_=kt_[order]
            uniq,first=np.unique(ksid,return_index=True)
            pe=np.add.reduceat(kpe,first); tt=np.minimum.reduceat(kt_,first)
        else:
            uniq=np.array([],np.uint16); pe=np.array([],np.float32); tt=np.array([],np.float32)
        sg=Sw.create_group(en)
        sg.create_dataset("sensor_idx",data=uniq.astype(np.uint16),**GZ)
        sg.create_dataset("PE",data=pe.astype(np.float32),**GZ); sg.create_dataset("T",data=tt.astype(np.float32),**GZ)
        sg.attrs["source_event_idx"]=np.uint32(src_idx); sg.attrs["n_hits"]=np.int64(int(uniq.size))
        out_src_idx.append(src_idx); out+=1

# fix config source_event_idx + n_events on all four
arr=np.array(out_src_idx,dtype=np.uint32)
for W in (Sw,Hw,Gw,Lw):
    if "config" in W:
        if "source_event_idx" in W["config"]: del W["config"]["source_event_idx"]
        W["config"].create_dataset("source_event_idx",data=arr)
        W["config"].attrs["n_events"]=np.int64(out)
So.close();Ho.close();Go.close();Lo.close()
for W in (Sw,Hw,Gw,Lw): W.close()
for P in (sP,hP,gP,lP): os.replace(P+".part",P)
print(f"[{tag}] {len(src_events)} src events -> {out} single-interaction events")
