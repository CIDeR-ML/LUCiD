// Refuse a time-PDF parameterisation that fittpdf wrote but fiTQun cannot use.
// fittpdf writes a full-size file and exits 0 even when every coefficient is NaN,
// and finiteness alone is not enough: a momentum grid too sparse for the
// hardcoded 9th-order momentum polynomial (ntpdfppar=10) produces a finite but
// oscillating fit whose width goes negative, which reads as a slow fit rather
// than a broken tune. gtcsgpar_0 IS the Gaussian width at 1 pe, so its sign is
// a direct physical check.
void check_tpdfpar(const char* fn, int minnode=18){
  TFile *f = TFile::Open(fn);
  if(!f || f->IsZombie()){ printf("TPDFPAR_BAD: cannot open %s\n", fn); return; }
  TIter nx(f->GetListOfKeys()); TKey* k; int ngraph=0, nbad=0, nneg=0, nnode=-1;
  while((k=(TKey*)nx())){
    TString n=k->GetName();
    if(!(n.BeginsWith("gtcmnpar")||n.BeginsWith("gtcsgpar"))) continue;
    TGraph* g=(TGraph*)k->ReadObj(); ngraph++;
    if(nnode<0 || g->GetN()<nnode) nnode=g->GetN();
    int bad=0;
    for(int i=0;i<g->GetN();i++){ double x,y; g->GetPoint(i,x,y);
      if(TMath::IsNaN(y)||!TMath::Finite(y)) bad++; }
    if(bad){ printf("  %s: %d/%d non-finite\n", n.Data(), bad, g->GetN()); nbad++; }
    if(n=="gtcsgpar_0"){
      int neg=0; double ymin=1e30;
      for(int i=0;i<g->GetN();i++){ double x,y; g->GetPoint(i,x,y);
        if(y<ymin) ymin=y; if(y<=0.) neg++; }
      if(neg){ printf("  %s: %d/%d nodes with width<=0 (min %.4g ns)\n",
                      n.Data(), neg, g->GetN(), ymin); nneg+=neg; }
    }
  }
  if(ngraph==0){ printf("TPDFPAR_BAD: no gtcmnpar/gtcsgpar graphs found\n"); return; }
  if(nbad){ printf("TPDFPAR_BAD: %d of %d coefficient graphs non-finite\n", nbad, ngraph); return; }
  if(nneg){ printf("TPDFPAR_BAD: time-PDF width is negative at %d momentum nodes\n", nneg); return; }
  if(nnode<minnode){
    printf("TPDFPAR_BAD: only %d momentum nodes for a 9th-order momentum fit "
           "(need >=%d); the fit is underconstrained\n", nnode, minnode); return; }
  printf("TPDFPAR_OK: %d coefficient graphs, %d momentum nodes, all finite, width positive\n",
         ngraph, nnode);
}
