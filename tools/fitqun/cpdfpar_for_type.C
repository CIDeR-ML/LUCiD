// Turn the reference charge-PDF merge output into the per-PMT-type file fiTQun
// actually loads.
//
//   root -l -b -q 'cpdfpar_for_type.C("cPDFpar.root", 0, "cPDFpar_PMT20inch.root")'
//
// Why this exists: MakecPDFparFile.cc writes both photosensor types into one
// file and keeps the type in every object name --
//
//     hCPDFrange_type0, gParam_type0_Rang3_2, gmuthr_type0_Rang3_1, hPunhitPar_type0
//
// while fQChrgPDF::LoadParams reads UN-suffixed names and is called once per
// type with a different file each time (fQChrgPDF.cc:55-79):
//
//     hCPDFrange,        gParam_Rang3_2,       gmuthr_Rang3_1,       hPunhitPar
//
// Loading the merge output directly therefore finds nothing and segfaults on the
// first unguarded dereference, which is what an empty cPDFpar does too. This
// copies one type's objects out under the names the loader wants.
void cpdfpar_for_type(const char* in_name, int iPMTType, const char* out_name) {
  TFile in(in_name);
  if (in.IsZombie()) { printf("cannot open %s\n", in_name); return; }

  TH1D* range = (TH1D*)in.Get(Form("hCPDFrange_type%d", iPMTType));
  if (!range) { printf("no hCPDFrange_type%d in %s\n", iPMTType, in_name); return; }

  TFile out(out_name, "RECREATE");
  range->Write("hCPDFrange");

  const int nRang = range->GetXaxis()->GetNbins();
  int nGraphs = 0;
  // The loader runs iRang from 0 to nRang INCLUSIVE, so the extra range past the
  // last bin is copied too; omitting it leaves a null TGraph at load.
  for (int iRang = 0; iRang <= nRang; iRang++) {
    const int nparam = (int)(range->GetBinContent(iRang + 1) + 1e-8);
    for (int i = 0; i < nparam; i++) {
      TGraph* g = (TGraph*)in.Get(Form("gParam_type%d_Rang%d_%d", iPMTType, iRang, i));
      if (!g) { printf("missing gParam_type%d_Rang%d_%d\n", iPMTType, iRang, i); continue; }
      out.cd(); g->Write(Form("gParam_Rang%d_%d", iRang, i)); nGraphs++;
    }
    for (int k = 0; k < 2; k++) {
      TGraph* g = (TGraph*)in.Get(Form("gmuthr_type%d_Rang%d_%d", iPMTType, iRang, k));
      if (!g) { printf("missing gmuthr_type%d_Rang%d_%d\n", iPMTType, iRang, k); continue; }
      out.cd(); g->Write(Form("gmuthr_Rang%d_%d", iRang, k)); nGraphs++;
    }
  }

  if (TH1D* p = (TH1D*)in.Get(Form("hPunhitPar_type%d", iPMTType)))
    { out.cd(); p->Write("hPunhitPar"); }
  else printf("note: no hPunhitPar_type%d (optional, the loader guards it)\n", iPMTType);

  if (TH2D* h = (TH2D*)in.Get(Form("hst2d_type%d", iPMTType)))
    { out.cd(); h->Write(Form("hst2d_type%d", iPMTType)); }

  out.Close();
  printf("wrote %s: %d ranges, %d graphs, from type %d\n",
         out_name, nRang, nGraphs, iPMTType);
}
