// combine all 1d charge distribution into a 2d histogram, i.e. f(q|mu)
// Obtain coefficients for P_unhit

#include <iostream>
#include <fstream>

#include <TH1D.h>
#include <TH2D.h>
#include <TFile.h>
#include <TF1.h>
#include <TGraphAsymmErrors.h>
#include <TMath.h>

using namespace std;

int gen2d(){
  
  const int nmumax=500;
  Double_t mu[nmumax];
  char mustr[nmumax][10];
  int nmuval;
  
  ifstream fin("mutbl.txt");
  
  double mutmp=0;
  int i=0;
  while ( 1 ) {
    if (i>=nmumax-1) {
      cout << "Array is too small!" << endl;
      exit(-1);
    }
    fin >> mustr[i];
    
    mu[i]=atof(mustr[i]);
    if (!(mutmp<mu[i])) {
      cout << "mutbl.txt: mu must be in ascending order!" << endl;
      break;
      //exit(-1);
    }
    cout << mustr[i] << endl;
    
    mutmp=mu[i];
    i++;
    
    if (fin.eof()) break;
  }
  nmuval=i;
  fin.close();
  
  mu[nmuval]=2.*mu[nmuval-1]-mu[nmuval-2];
  
  TH1D *hnHit[2],*hnTot[2];
  for (int iPMTType=0; iPMTType<2; iPMTType++) {// Use lower bin edge!
    hnHit[iPMTType] = new TH1D(Form("hnHit_type%d",iPMTType),"Total # of hits",nmuval,mu);
    hnTot[iPMTType] = new TH1D(Form("hnTot_type%d",iPMTType),"Total # of active PMTs",nmuval,mu);
  }
  
  char cPath[256];
  strcpy(cPath,gDirectory->GetPath());
  
  int nqbns;
  TH2D *h2d[2]={NULL,NULL};;
  
  for (i=0; i<nmuval; i++) {
    TFile *hstf = new TFile(Form("%s_pdf.root",mustr[i]));
    gDirectory->cd(cPath);
    
    TH1D *hctr = (TH1D*)(hstf->Get("hctr"));
    
    for (int iPMTType=0; iPMTType<2; iPMTType++) {
      TH1D *h1dtmp = (TH1D*)(hstf->Get(Form("hchpdf%d",iPMTType+2)));
      h1dtmp->Sumw2();
      
      hnHit[iPMTType]->SetBinContent(i+1,h1dtmp->GetEntries());// Total # of hits
      hnTot[iPMTType]->SetBinContent(i+1,hctr->GetBinContent(iPMTType+2));// Total # of active PMTs
      
      // Normalized PDF!
      h1dtmp->Scale(1./h1dtmp->GetEntries(),"width");
      
      if (h2d[iPMTType]==NULL) {
        nqbns=h1dtmp->GetNbinsX();
        Double_t *bEdgs = new Double_t[nqbns+1];
        for (int j=0; j<=nqbns; j++) {
          bEdgs[j]=h1dtmp->GetBinLowEdge(j+1);
        }
        h2d[iPMTType] = new TH2D(Form("hst2d_type%d",iPMTType),"Charge PDF f(q|#mu)",nmuval,mu,nqbns,bEdgs);
        h2d[iPMTType]->GetXaxis()->SetTitle("Predicted charge #mu (p.e.)");
        h2d[iPMTType]->GetYaxis()->SetTitle("Observed charge q (p.e.)");
        h2d[iPMTType]->SetOption("COLZ");
        
        delete[] bEdgs;
      }
      
      for (int j=1; j<=nqbns+1; j++) {// make sure to fill overflow bin!
        h2d[iPMTType]->SetBinContent(i+1,j,h1dtmp->GetBinContent(j));
        h2d[iPMTType]->SetBinError(i+1,j,h1dtmp->GetBinError(j));
      }
      
    }
    
    delete hstf;
  }
  
  TFile ofile("pdf2d.root", "RECREATE");
  
  for (int iPMTType=0; iPMTType<2; iPMTType++) {
    h2d[iPMTType]->Write();
    
    hnHit[iPMTType]->Write();
    hnTot[iPMTType]->Write();
  }
  
  const double muFitRange = 200.;
  
  for (int iPMTType=0; iPMTType<2; iPMTType++) {
    TH1D *hPunhitPar = new TH1D(Form("hPunhitPar_type%d",iPMTType),"c_n for P(unhit|#mu)",10,0.5,10.5);
    std::cout << "DEBUG ==========-- Creates unhitpar histo" << std::endl;
    if (hnTot[iPMTType]->Integral()>0.) {

      TGraphAsymmErrors *gPHit = new TGraphAsymmErrors(hnHit[iPMTType],hnTot[iPMTType]);
      std::cout << "DEBUG ==========-- hnTot integral is more than zero, creates gPHit" << std::endl;
      std::cout << "DEBUG ==========-- gPHit N " << gPHit->GetN() << std::endl;
      gPHit->SetName(Form("gPHit_type%d",iPMTType));
      gPHit->SetTitle(Form("P(hit|#mu) for Type %d PMT",iPMTType));
      
      for (i=0; i<nmuval-1; i++) {
	std::cout << "DEBUG ========-- loops nmmuval " << i << std::endl;
        gPHit->GetX()[i] = mu[i];// mu value is exactly at lower bin edge
	std::cout << "DEBUG ========-- sets ith x value to  " <<mu[i] << std::endl;
        gPHit->SetPointEXlow(i,0.);
	std::cout << "DEBUG ========-- sets EXLoq " << std::endl;
        gPHit->SetPointEXhigh(i,0.);
	std::cout << "DEBUG ========-- sets EXHigh " << std::endl;
	std::cout << "i " << i << "  nmuval " << nmuval << std::endl;
      }
      
      std::cout << "DEBUG ========-- ends nmuval loop" << std::endl;

      //      TF1 *fPhit = new TF1(Form("fPhit_type%d",iPMTType),"(1.-exp(-x))++x*exp(-x)++x*x*exp(-x)++x*x*x*exp(-x)",0.,muFitRange);
      TF1 *fPhit = new TF1(Form("fPhit_type%d",iPMTType),"(1.-[0]*exp(-x))+[1]*x*exp(-x)+[2]*x*x*exp(-x)+[3]*x*x*x*exp(-x)",0.,muFitRange);
      fPhit->FixParameter(0,1.);
      fPhit->SetParameter(1,-0.);
      fPhit->SetParameter(2,-0.);
      fPhit->SetParameter(3,-0.);
      for (int k = 1; k<=3; k++) fPhit->SetParLimits(k, -1/TMath::Factorial(k), 0.); //yoshida
       

      std::cout << "DEBUG ========-- Declares TF1" << std::endl;
      gPHit->Fit(fPhit,"VR");
      std::cout << "DEBUG ========-- ends fit" << std::endl;
      
      cout << Form("P_unhit coefficients for Type %d PMT:",iPMTType) << endl;
      
      for (int k=1; k<=3; k++) {

        double c_k = -(fPhit->GetParameter(k));

        hPunhitPar->SetBinContent(k,c_k);
        double p_k = c_k*TMath::Factorial(k);
        cout << Form("  c_%d = %f, p_%d = %f",k,c_k,k,p_k) << endl;

        if (!(p_k+1e-3>=0. && p_k<=1.-1e-3)) { //yoshida

	  std::cout << "ERROR: Bad Phit fit. k = " << k << " p_k = " << p_k << std::endl;
	  exit(-1);

	}

      }
      
      std::cout << "DEBUG ==========-- just before writing gPHit! iPMTType: " << iPMTType << std::endl;
      gPHit->Write();
      std::cout << "DEBUG ==========-- Wrote gPHit! iPMTType: " << iPMTType << std::endl;
    }
    hPunhitPar->Write();
  }
  
  ofile.Close();// Close the file
  return 0;
}

//Double_t fphit(Double_t *x, Double_t *par){
//  float mu=x[0];
//  Double_t tmp=1.-(1+par[0]*mu+par[1]*pow(mu,2)+par[2]*pow(mu,3))*exp(-mu);
//  return tmp;
//
