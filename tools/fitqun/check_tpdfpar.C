// Refuse a time-PDF parameterisation whose coefficient graphs are not finite.
// fittpdf writes a full-size file and exits 0 even when every coefficient is NaN.
void check_tpdfpar(const char* fn){
  TFile *f = TFile::Open(fn);
  if(!f || f->IsZombie()){ printf("TPDFPAR_BAD: cannot open %s\n", fn); return; }
  TIter nx(f->GetListOfKeys()); TKey* k; int ngraph=0, nbad=0;
  while((k=(TKey*)nx())){
    TString n=k->GetName();
    if(!(n.BeginsWith("gtcmnpar")||n.BeginsWith("gtcsgpar"))) continue;
    TGraph* g=(TGraph*)k->ReadObj(); ngraph++;
    int bad=0;
    for(int i=0;i<g->GetN();i++){ double x,y; g->GetPoint(i,x,y);
      if(TMath::IsNaN(y)||!TMath::Finite(y)) bad++; }
    if(bad){ printf("  %s: %d/%d non-finite\n", n.Data(), bad, g->GetN()); nbad++; }
  }
  if(ngraph==0){ printf("TPDFPAR_BAD: no gtcmnpar/gtcsgpar graphs found\n"); return; }
  if(nbad){ printf("TPDFPAR_BAD: %d of %d coefficient graphs non-finite\n", nbad, ngraph); return; }
  printf("TPDFPAR_OK: %d coefficient graphs, all finite\n", ngraph);
}
