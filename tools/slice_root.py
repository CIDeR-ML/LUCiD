"""Cut the first N events of a PhotonSim ROOT file into a small file, every branch verbatim.

    python tools/slice_root.py SRC DST --events N

The example events in the ``lucid-data`` release asset are cut with this (see
``tools/assets/lucid-data-v1.manifest``). It needs PyROOT, which develop.sif has: uproot cannot
write a ``std::vector<std::string>`` branch, and ``TrackInfo_CreatorProcess`` is one.

``OpticalPhotons`` holds one entry per event and keeps entries ``0..N-1``. ``OpticalPhotonsRaw``
holds the photons in chunks, several entries per event, each stamped with its ``EventID``; PhotonSim
writes them in event order, so the chunks with ``EventID < N`` are its leading entries, and only
those are read (a selection would read the whole multi-GB tree). The histograms are copied whole. A reader resolves a bare
tree name to its highest cycle, and the slice carries that cycle only.
"""
import argparse
import os

import ROOT


def _leading_chunks(raw, n_events):
    """How many leading ``OpticalPhotonsRaw`` entries hold events ``0..n_events-1``."""
    raw.SetBranchStatus('*', 0)
    raw.SetBranchStatus('EventID', 1)
    ids = []
    for i in range(raw.GetEntries()):
        raw.GetEntry(i)
        ids.append(raw.EventID)
    raw.SetBranchStatus('*', 1)
    n = sum(1 for e in ids if e < n_events)
    if any(e >= n_events for e in ids[:n]):
        raise ValueError(f'the chunks of events 0-{n_events - 1} are not the leading entries')
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('src', help='PhotonSim ROOT file')
    ap.add_argument('dst', help='the slice to write')
    ap.add_argument('--events', type=int, required=True, help='keep events 0..N-1')
    a = ap.parse_args()

    src = ROOT.TFile.Open(a.src)
    part = a.dst + '.part'
    dst = ROOT.TFile(part, 'RECREATE')
    for name in sorted({key.GetName() for key in src.GetListOfKeys()}):
        obj = src.Get(name)
        dst.cd()
        if name == 'OpticalPhotonsRaw':
            obj.CopyTree('', '', _leading_chunks(obj, a.events), 0).Write()
        elif obj.InheritsFrom('TTree'):
            obj.CopyTree('', '', a.events, 0).Write()
        else:
            obj.Write(name)
    dst.Purge()                # drop the intermediate autosave cycles of a large tree
    dst.Close()
    src.Close()
    os.replace(part, a.dst)
    print(f'wrote {a.dst} ({os.path.getsize(a.dst) / 1e6:.1f} MB, events 0-{a.events - 1})')


if __name__ == '__main__':
    main()
