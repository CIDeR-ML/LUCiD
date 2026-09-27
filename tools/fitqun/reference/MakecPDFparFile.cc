// Script to produce the final charge pdf root file which fiTQun reads

int MakecPDFparFile(){
  
  // Create the final charge pdf file!
  TFile fout("cPDFpar.root","RECREATE");
  
  for (int iPMTType=0; iPMTType<2; iPMTType++) {
    
    TFile *parf= new TFile(Form("fitpdf_type%d.root",iPMTType));
    
    fout.cd();
    
    TH1D *hCPDFrange = (TH1D*)(parf->Get(Form("hCPDFrange_type%d",iPMTType)));
    hCPDFrange->Write();
    
    int nRang = hCPDFrange->GetXaxis()->GetNbins();
    
    for (int iRang=0; iRang<=nRang; iRang++) {
      int nparam = (int)(hCPDFrange->GetBinContent(iRang+1)+1e-8);
      
      for (int i=0; i<nparam; i++) {
        ((TGraph*)(parf->Get(Form("gParam_type%d_Rang%d_%d",iPMTType,iRang,i))))->Write();
      }
      
      for (int k=0; k<2; k++) {
        ((TGraph*)(parf->Get(Form("gmuthr_type%d_Rang%d_%d",iPMTType,iRang,k))))->Write();
      }
    }
    
    ((TH2D*)(parf->Get(Form("hst2d_type%d",iPMTType))))->Write();
    
    ((TH1D*)(parf->Get(Form("hPunhitPar_type%d",iPMTType))))->Write();
  }
  
  fout.Close();

  return 0;
}
