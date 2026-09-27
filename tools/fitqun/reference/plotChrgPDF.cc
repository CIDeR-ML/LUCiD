
void plotChrgPDF(int flgLogxy=0, int flgLogz=1) {// define threshold for mu and output in file, to be used by fitpdf
  
  TFile *_file0 = TFile::Open("pdf2d.root");
  
  TH2D *htmp=(TH2D*)_file0->Get("hst2d_type0");
  
  TString ifName(_file0->GetName());
  ifName.ReplaceAll(".root","");
  
  TGraph *gHi = new TGraph();
  gHi->SetName("gHi");
  TGraph *gLo = new TGraph();
  gLo->SetName("gLo");
  
  int nqbins = htmp->GetNbinsY();
  
  for (int iqbin=0; iqbin<nqbins; iqbin++) {// loop over q bins
    double qval = htmp->GetYaxis()->GetBinCenter(iqbin+1);
    
    // define functions for thresholds
    TH1D *h1dmu = htmp->ProjectionX("proj",iqbin+1,iqbin+1);
    double muthrLo,muthrHi;
    
    if (0) {
      Double_t parr[3]={0.01,0.5,0.99};
      Double_t qarr[3];
      h1dmu->GetQuantiles(3,qarr,parr);
      muthrLo = qarr[0];
      muthrHi = qarr[2];
    }
    else {
      double thresh_z=h1dmu->GetMaximum()/100.;
      muthrLo = h1dmu->GetXaxis()->GetBinLowEdge(h1dmu->FindFirstBinAbove(thresh_z));
      muthrHi = h1dmu->GetXaxis()->GetBinLowEdge(h1dmu->FindLastBinAbove(thresh_z));
      
      double mu_mean = h1dmu->GetMean();
//      muthrLo = 1.2*(muthrLo-mu_mean)+mu_mean;
//      muthrHi = 1.2*(muthrHi-mu_mean)+mu_mean;
    }
    
//    muthrLo = fthr_Lo(qval);
//    muthrHi = fthr_Hi(qval);
    
    gHi->SetPoint(iqbin,muthrHi,qval);
    gLo->SetPoint(iqbin,muthrLo,qval);
  }
  
  TH1D *hDum = new TH1D("hDum","",10,0,10);
  
  int nSmooth=3;// smooth the graphs!
  
  hDum->SmoothArray(nqbins,gHi->GetX(),nSmooth);
  hDum->SmoothArray(nqbins,gLo->GetX(),nSmooth);
  
  gStyle->SetOptStat(0);
  
  htmp->Draw("COLZ");
  
  gHi->SetLineWidth(2);
  gLo->SetLineWidth(2);
  gHi->Draw("LSAME");
  gLo->Draw("LSAME");
  
  TString strLogxy="";
  if (flgLogxy) {
    gPad->SetLogx(1);
    gPad->SetLogy(1);
    strLogxy="_logxy";
  }
  
  TString strLogz="";
  if (flgLogz) {
    gPad->SetLogz(1);
    strLogz="_logz";
  }
  
  
  TFile *fThrIn = new TFile("muthresh.root","RECREATE");
  
  htmp->Write();
  
  gHi->Write();
  gLo->Write();
  
  TGraph *gmuthr_Hi = new TGraph(gHi->GetN(),gHi->GetY(),gHi->GetX());
  gmuthr_Hi->SetName("gmuthr_Hi");
  TGraph *gmuthr_Lo = new TGraph(gLo->GetN(),gLo->GetY(),gLo->GetX());
  gmuthr_Lo->SetName("gmuthr_Lo");
  
  gmuthr_Hi->Write();
  gmuthr_Lo->Write();
  
  fThrIn->Close();
  
  TCanvas *c1 = (TCanvas*)gPad; if (!c1) c1 = new TCanvas("c1","c1");
  c1->Print(Form("plt_%s_%s%s%s.pdf",ifName.Data(),htmp->GetName(),strLogxy.Data(),strLogz.Data()));

}

//double fthr_Lo(double qval) {
//  
//  double thrw=sqrt(qval)*4.;
//  
//  return qval-thrw;
//}

//double fthr_Hi(double qval) {
//  
//  double thrw=sqrt(qval)*4.;
//  
//  return qval+thrw;
//}

