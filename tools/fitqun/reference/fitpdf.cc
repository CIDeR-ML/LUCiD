
// fit f(q|mu) at different q range

//#include <iostream>
//
//#include <TF1.h>
//#include <TGraph.h>
//#include <TGraphErrors.h>
//#include <TGraphAsymmErrors.h>
//#include <TH1.h>
//#include <TH1D.h>
//#include <TH2D.h>
//#include <TFile.h>

bool flgUseInputThrGraph=true;
TGraph *gmuThrIn[2];

//iPMTType= 0:old, 1:new PMT
void GetmuThresh(double qval, double &muthrLo, double &muthrHi);

int fitpdf(int iPMTType, int iSK_Ver){
  
  gROOT->ProcessLine(".L fQChrgPDF.cc+");
  
  fQChrgPDF::Get()->SetGlobal_PMTType(iPMTType);
  
//  fQChrgPDF::Get()->flgStrict = true;
  
  if (flgUseInputThrGraph) {
    TFile *fThrIn = new TFile("muthresh.root");
    if (fThrIn->IsZombie()) {
      flgUseInputThrGraph=false;
    }
    else {
      cout << "Using input mu threshold graphs!" << endl;
      gmuThrIn[0] = (TGraph*)(fThrIn->Get("gmuthr_Lo"));
      gmuThrIn[1] = (TGraph*)(fThrIn->Get("gmuthr_Hi"));
    }
  }
  
//  char cPath[256];
//  strcpy(cPath,gDirectory->GetPath());
  TFile *hstf= new TFile("pdf2d.root");
//  gDirectory->cd(cPath);
  
  TH2D *hpdf2d=(TH2D*)(hstf->Get(Form("hst2d_type%d",iPMTType)));
  int nbnsY=hpdf2d->GetNbinsY();
  cout << "# of q bins: " << nbnsY << endl;
  
  // Define range of q
  const int nqRang = 3;// # of q range (excl. overflow bin)
  double qRang[nqRang+1]={0.,1.45,29.5,1000.};// q range boundaries; each boundary has to be exactly at a bin center!
  const int nparam_Arr[nqRang+1]={4,6,4,6};// # of parameters in each q range(+ overflow bin)
  
  //  const double muthrHi_Satu[4]={250.,220.,220.,980.};// upper mu threshold for saturation bin for each SK era
  //  const double qSatu[4]={161.,155.,155.,802.5};// saturation value for each SK era
  const double muthrHi_Satu[4]={250.,220.,220.,1250.};// upper mu threshold for saturation bin for each SK era
  const double qSatu[4]={161.,155.,155., 1225.};// saturation value for each SK era
  //  const double muthrHi_Satu[4]={1250.,220.,220.,500.};// upper mu threshold for saturation bin for each SK era
  //  const double qSatu[4]={1250.,155.,155.,500.};// saturation value for each SK era
  qRang[nqRang] = qSatu[iSK_Ver-1];
  
  cout << Form("SK %d: q_satu=%f",iSK_Ver,qRang[nqRang]) << endl;
  cout << Form("Fitting charge PDF for Type %d PMT",iPMTType) << endl;
  
  fQChrgPDF::Get()->hCPDFrange[iPMTType] = new TH1D(Form("hCPDFrange_type%d",iPMTType),"",nqRang,qRang);
  for (int iRang=0; iRang<=nqRang; iRang++) {// also set overflow bin
    fQChrgPDF::Get()->hCPDFrange[iPMTType]->SetBinContent(iRang+1,nparam_Arr[iRang]);
    
    for (int i=0; i<nparam_Arr[iRang]; i++) {
      fQChrgPDF::Get()->gParam[iPMTType][iRang][i] = new TGraph();
      fQChrgPDF::Get()->gParam[iPMTType][iRang][i]->SetName(Form("gParam_type%d_Rang%d_%d",iPMTType,iRang,i));
    }
    
    for (int k=0; k<2; k++) {
      fQChrgPDF::Get()->gmuthr[iPMTType][iRang][k] = new TGraph();
      fQChrgPDF::Get()->gmuthr[iPMTType][iRang][k]->SetName(Form("gmuthr_type%d_Rang%d_%d",iPMTType,iRang,k));
    }
  }
  
  int nqBin = hpdf2d->GetYaxis()->FindBin(fQChrgPDF::Get()->hCPDFrange[iPMTType]->GetBinLowEdge(nqRang+1));// bin # of cutoff bin
  cout << "# of q bins in range: " << nqBin << endl;
  
  TFile *ofile= new TFile(Form("fitpdf_type%d.root",iPMTType), "RECREATE");
  
  const double mumax=1200.;
  
  TF1 *flgcPDF = new TF1("flgcPDF",fQChrgPDF::flogcPDFptr,0.,mumax,fQChrgPDF::nParamMax);
  flgcPDF->SetNpx(10000);
  
  cout << "nParamMax = " << flgcPDF->GetNpar() << endl;
  
  int bufRang=-1;
  for (int k=1; k<=nqBin+1; k++) {// do the boundary twice!
    
    int iRang = fQChrgPDF::Get()->hCPDFrange[iPMTType]->GetXaxis()->FindBin(hpdf2d->GetYaxis()->GetBinLowEdge(k));
    iRang--;// c++ indexing!
    
    if (bufRang!=iRang && k>1) {// change of q range!
      cout << Form("End of processing iRang=%d",bufRang) << endl;
      k--;// Process the boundary bin again with incremented iRang
    }
    
    cout << "################################################" << endl;
    
    int nparam = nparam_Arr[iRang];//# of parameters to fit
    
    double qval = hpdf2d->GetYaxis()->GetBinCenter(k);// q value of this bin
    
    if (bufRang!=iRang) {// slightly offset q to lie on proper range
      fQChrgPDF::Get()->SetGlobal_qval(qval+1e-8);
    }
    else {
      fQChrgPDF::Get()->SetGlobal_qval(qval-1e-8);
    }
    
    cout << Form("k=%d: Observed q=%f",k,qval) << endl;
    // Self-describing progress: index AND total, flushed, so any watcher
    // can quote a measured ETA without being told how long the loop is.
    cout << "PROGRESS " << k << "/" << (nqBin+1) << std::endl << std::flush;
    cout << Form(" iRang=%d, nparam=%d",iRang,nparam) << endl;
    
    // threshold of mu, outside of which is gaussian #############
    double muthrLo,muthrHi;
    GetmuThresh(qval,muthrLo,muthrHi);
    
    if (iRang>=nqRang) muthrHi=muthrHi_Satu[iSK_Ver-1];
    
    fQChrgPDF::Get()->gmuthr[iPMTType][iRang][0]->SetPoint(fQChrgPDF::Get()->gmuthr[iPMTType][iRang][0]->GetN(),qval,muthrLo);
    fQChrgPDF::Get()->gmuthr[iPMTType][iRang][1]->SetPoint(fQChrgPDF::Get()->gmuthr[iPMTType][iRang][1]->GetN(),qval,muthrHi);
    
    // ###########################################################
    
    // produce TGraph's for fitting as function of mu
    
    int nmuBin = hpdf2d->GetNbinsX();
    Double_t *gEntX = new Double_t[nmuBin];
    Double_t *gEntY = new Double_t[nmuBin];
    Double_t *gEntYlg = new Double_t[nmuBin];
    Double_t *gErr = new Double_t[nmuBin];
    Double_t *gErru = new Double_t[nmuBin];
    Double_t *gErrd = new Double_t[nmuBin];
    
    int ngent=0;
    for (int i=0; i<nmuBin; i++) {
      Double_t bEdgs = hpdf2d->GetXaxis()->GetBinLowEdge(i+1);// mu is set at lower edge
      
      Double_t bCont,bErr;
      if (iRang>=nqRang) {// overflow bin, integrate all bins above!
        bCont = hpdf2d->IntegralAndError(i+1,i+1,k,-1,bErr,"width");
        // original histogram is not density in mu!
        bCont /= hpdf2d->GetXaxis()->GetBinWidth(i+1);
        bErr  /= hpdf2d->GetXaxis()->GetBinWidth(i+1);
      }
      else {
        bCont = hpdf2d->GetBinContent(i+1,k);
        bErr = hpdf2d->GetBinError(i+1,k);
      }
      
      if (bCont>0.) {
//        cout << bErr/bCont << endl;
        gEntX[ngent] = bEdgs;
        gEntY[ngent] = bCont;
        gEntYlg[ngent] = log(bCont);
        gErr[ngent] = bErr;
        gErru[ngent] = log(1+bErr/bCont);
        Double_t Errtmp = 1-bErr/bCont;
        if (!(Errtmp>0.)) Errtmp=1e-100; 
        gErrd[ngent] = -log(Errtmp);
        ngent++;
      }
    }
    TGraphAsymmErrors *glogPDF = new TGraphAsymmErrors(ngent,gEntX,gEntYlg,NULL,NULL,gErrd,gErru);
    TGraphErrors *gPDF = new TGraphErrors(ngent,gEntX,gEntY,NULL,gErr);
    
    glogPDF->SetName(Form("glogPDF_type%d_Rang%d_%d",iPMTType,iRang,k));
    gPDF->SetName(Form("gPDF_type%d_Rang%d_%d",iPMTType,iRang,k));
    
    // ###########################################################
    
    if (bufRang!=iRang) {// change of q range! - seed with gaussian
//      TF1 *fprefit = new TF1("fprefit","1++x++x*x",0.,mumax);// seed with gaussian fit
      TF1 *fprefit = new TF1("fprefit","1++x++x*x",fQChrgPDF::Get()->gmuthr[iPMTType][iRang][0]->Eval(qval),fQChrgPDF::Get()->gmuthr[iPMTType][iRang][1]->Eval(qval));// seed with gaussian fit
      
      glogPDF->Fit(fprefit,"QR","");
      
      flgcPDF->SetParameter(0,fprefit->GetParameter(0));
      flgcPDF->SetParameter(1,fprefit->GetParameter(1));
      flgcPDF->SetParameter(2,fprefit->GetParameter(2));
      for (int i=3; i<nparam; i++) {
        flgcPDF->SetParameter(i,0);
      }
      
      for (int i=0; i<nparam; i++) {
        flgcPDF->ReleaseParameter(i);
      }
      for (int i=nparam; i<flgcPDF->GetNpar(); i++) {
        flgcPDF->FixParameter(i,0);
      }
      
      delete fprefit;
    }
    
    cout << "Initial parameters:" << endl;
    for (int j=0; j<nparam; j++) {
      cout << j << ": " << flgcPDF->GetParameter(j) << endl;
    }
    
    glogPDF->Fit(flgcPDF,"Q","",0,mumax);
    
    // Save the fit parameters!
    cout << "Fit parameters:" << endl;
    for (int j=0; j<nparam; j++) {
      fQChrgPDF::Get()->gParam[iPMTType][iRang][j]->SetPoint(fQChrgPDF::Get()->gParam[iPMTType][iRang][j]->GetN(),qval,flgcPDF->GetParameter(j));
      cout << j << ": " << flgcPDF->GetParameter(j) << endl;
    }
    
    glogPDF->Write();
    gPDF->Write();
    
    delete[] gEntX;
    delete[] gEntY;
    delete[] gEntYlg;
    delete[] gErr;
    delete[] gErru;
    delete[] gErrd;
    
    bufRang=iRang;
    
    if (iRang>=nqRang) {
      break;
    }
    
  }
  
  
  fQChrgPDF::Get()->hCPDFrange[iPMTType]->Write();
  for (int iRang=0; iRang<=nqRang; iRang++) {
    for (int i=0; i<nparam_Arr[iRang]; i++) {
      fQChrgPDF::Get()->gParam[iPMTType][iRang][i]->Write();
    }
    
    for (int k=0; k<2; k++) {
      fQChrgPDF::Get()->gmuthr[iPMTType][iRang][k]->Write();
    }
  }
  
  if (flgUseInputThrGraph) {
    gmuThrIn[0]->Write();
    gmuThrIn[1]->Write();
  }
  
  hpdf2d->Write();
  ((TH1D*)(hstf->Get(Form("hPunhitPar_type%d",iPMTType))))->Write();
  
  ofile->Close();// Close the file
  
}

void GetmuThresh(double qval, double &muthrLo, double &muthrHi) {// Define thresholds of mu
  
  double thrw=sqrt(qval)*4.;
  muthrLo=qval-thrw;
  muthrHi=qval+thrw;
  
  if (flgUseInputThrGraph) {
    muthrLo = gmuThrIn[0]->Eval(qval);
    muthrHi = gmuThrIn[1]->Eval(qval);
  }
  

  return 0;
}
