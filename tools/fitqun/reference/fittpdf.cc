//script to normalize the cherenkov profiles and put them into one file

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <TH1D.h>
#include <TH2D.h>
#include <TH3D.h>
#include <TGraphErrors.h>
#include <TFile.h>
#include <TMath.h>

using namespace std;

const int nthrdpars=10;//number of momentum fit parameters
const int npars=7;//number of logmu fit parameters

bool shiftLeft = true;
bool useEMG = false;
bool useFitSlices = true;

Double_t emg(Double_t *x, Double_t *par){
  double xx = x[0];
  double k = par[0];
  double mu = par[1];
  double sigma = par[2];
  double lambda = par[3];
  
  return k*lambda/2 * TMath::Exp(lambda/2*(2*mu+lambda*pow(sigma,2)-2*xx)) * TMath::Erfc( (mu + lambda*pow(sigma,2) - xx) / (sqrt(2)*sigma) );
}

TString PolyFormula(int nord) {
  TString strPoly="1++x";
  for (int i=2; i<=nord; i++) {
    strPoly+=Form("++x^%d",i);
  }
  
  return strPoly;
}

int fittpdf(int PID, bool flogfit=false, bool fNoErrorbars=true){

  TFitResultPtr ptrres[2],ptrprres[2];
  const int nmommax=200;
  double momfitrang[2]={0.,7000.};
  double mnpar[npars],sgpar[npars],thdpr[2][npars][nthrdpars];
  
  TString strParNam[2]={"mn","sg"};
  
  Double_t armom[nmommax],arpar[2][npars][nmommax],arparerr[2][npars][nmommax];
  TH1D *hmnsg[2][nmommax];
  TH1D *hmeantmp,*hsigmtmp, *hchi2tmp;
  TH2D *hist2d;
  TGraphErrors *gtpar[2][npars];
  
  double pofst=0.;
  if (PID==11) {
    momfitrang[0]=20.;
    momfitrang[1]=5000.;
    pofst=-20.;
  }
  else if (PID==13) {
    momfitrang[0]=180.;
    momfitrang[1]=2000.;
    pofst=100.;
  }
  else if (PID==211) {
    momfitrang[0]=220.;
    momfitrang[1]=2000.;
    pofst=140.;
  }
  else if (PID==321) {
    momfitrang[0]=600.;
    momfitrang[1]=5000.;
    pofst=530.;
  }
  else if (PID==2212) {
    momfitrang[0]=1150.;
    momfitrang[1]=5000.;
    pofst=1020.;
  }
  
  if (flogfit) {
    momfitrang[0]=log(momfitrang[0]-pofst);
    momfitrang[1]=log(momfitrang[1]-pofst);
  }
  else {
    pofst=0.;
  }
  
  TFile *fitmp = new TFile(Form("%d_tpdfhist.root",PID));
  if (fitmp->IsZombie()) exit(-1);

  TH3D *h3d = (TH3D*)fitmp->Get("hist_tpdf");
  double tcmin=h3d->GetXaxis()->GetBinLowEdge(1);
  double tcmax=h3d->GetXaxis()->GetBinLowEdge(h3d->GetNbinsX()+1);

  TH2D * hmean_step0 = (TH2D*)h3d->Project3D("yz")->Clone("hmean_step0");
  TH2D * hmean_step1 = (TH2D*)h3d->Project3D("yz")->Clone("hmean_step1");

  TH2D * hsigm_step0 = (TH2D*)h3d->Project3D("yz")->Clone("hsigm_step0");
  TH2D * hsigm_step1 = (TH2D*)h3d->Project3D("yz")->Clone("hsigm_step1");
  
  TFitResultPtr timeResFitPointer;
  TF1 *myGausFit = new TF1("myGausfit","gaus", tcmin, tcmax);
  myGausFit->SetParName(0, "Const");
  myGausFit->SetParName(1, "Mean");
  myGausFit->SetParName(2, "Sigma");
  
  TF1 *fEMG = new TF1("fEMG",emg,tcmin,tcmax,4);
  fEMG->SetParName(0, "Const");
  fEMG->SetParName(1, "Mean");
  fEMG->SetParName(2, "Sigma");
  fEMG->SetParName(3, "Lambda");
  
  
  int nmom=h3d->GetNbinsZ();
  // Upstream forces 24 for mu/pi, which over-runs a histogram built from fewer
  // momentum points. Clamp to what is actually there.
  if (PID==13 || PID==211) nmom = std::min(nmom, 24);

  int ntmp=0;// hoisted: declared inside this loop upstream, used after it
  for (int k=0; k<nmom; k++) {
    int imom=(int)(h3d->GetZaxis()->GetBinLowEdge(k+1)+0.5);
    cout << imom << endl;
    armom[k]=imom;
    if (flogfit) armom[k]=log(armom[k]-pofst);
    h3d->GetZaxis()->SetRange(k+1,k+1);
    hist2d = (TH2D*)h3d->Project3D("yx");
    hist2d->FitSlicesX();
    hmeantmp = (TH1D*)gDirectory->Get("hist_tpdf_yx_1");
    hsigmtmp = (TH1D*)gDirectory->Get("hist_tpdf_yx_2");
    hchi2tmp = (TH1D*)gDirectory->Get("hist_tpdf_yx_chi2");
    ntmp=hmeantmp->GetNbinsX();

    if(useFitSlices){
      for (int muBin = 1; muBin <= ntmp; muBin++){
        hmean_step0->SetBinContent( k+1,muBin,hmeantmp->GetBinContent(muBin));
	hsigm_step0->SetBinContent( k+1,muBin,hsigmtmp->GetBinContent(muBin));
      }
    }else{
      // And now, let's fit again, but with sensible starting points
      for (int muBin = 1; muBin <= ntmp; muBin++){
	TH1D* hist1d = (TH1D*) hist2d->ProjectionX("projx", muBin, muBin);
	
	if (hist1d->GetEntries() == 0.) {
	  hmeantmp->SetBinContent(muBin, 0.);
	  hsigmtmp->SetBinContent(muBin, 0.);
	  hchi2tmp->SetBinContent(muBin, 0.);
	  continue;
	}
	
	myGausFit->SetParameter(0, 0.1);
	myGausFit->SetParameter(1, 0.);
	//      myGausFit->SetParLimits(1, tcmin*0.5, tcmax*0.5);
	myGausFit->SetParameter(2, 5.);

	if (useEMG){
	  fEMG->SetParameter(0, 0.1);
	  fEMG->SetParameter(1, 0.);
	  fEMG->SetParameter(2, 5.);
	  fEMG->SetParameter(3, 5.);
	}
	
	//	timeResFitPointer = hist1d->Fit("gaus", "S", "");
	///      timeResFitPointer = hist1d->Fit("myGausfit", "S", "",  hist1d->GetMean()-hist1d->GetRMS(), hist1d->GetMean()+hist1d->GetRMS());
	//	timeResFitPointer = hist1d->Fit("myGausfit", "S", "",  hist1d->GetRMS(), hist1d->GetRMS()/2);
	timeResFitPointer = hist1d->Fit("myGausfit", "S", "",  tcmin, tcmax);
	
	int goodFit = timeResFitPointer;
	
	if (shiftLeft == true && goodFit == 0){

	  double lowRange = tcmin;
	  double hiRange = timeResFitPointer->Parameter(1)+1.0*timeResFitPointer->Parameter(2);
	  
	  std::cout << "Shifting left. New fit range: " << lowRange << " " << hiRange << std::endl;
	  
	  timeResFitPointer = hist1d->Fit("myGausfit", "S", "", lowRange, hiRange);

	  goodFit = timeResFitPointer;
	}

	if (useEMG){
	  timeResFitPointer = hist1d->Fit("fEMG", "S", "", tcmin, tcmax);
	  goodFit = timeResFitPointer;
	}
	
	if (goodFit == 0){
	  hmeantmp->SetBinContent(muBin, timeResFitPointer->Parameter(1));
	  hsigmtmp->SetBinContent(muBin, timeResFitPointer->Parameter(2));
	  if (timeResFitPointer->Ndf()) hchi2tmp->SetBinContent(muBin, timeResFitPointer->Chi2()/timeResFitPointer->Ndf());
	  else  hchi2tmp->SetBinContent(muBin, 0.);
	} else {
	  
	  cout << "BAD STEP0 FIT " << muBin << " " << k << endl;
	  hmeantmp->SetBinContent(muBin, 0.);
	  hsigmtmp->SetBinContent(muBin, 0.);
	  hchi2tmp->SetBinContent(muBin, 0.);
	  
	  //	exit(-3);
	}
	
	hmean_step0->SetBinContent( k+1,muBin, 0.);
	hsigm_step0->SetBinContent( k+1,muBin, 0.);
	hmean_step1->SetBinContent( k+1,muBin, 0.);
	hsigm_step1->SetBinContent( k+1,muBin, 0.);
	
	if (goodFit == 0){
	  // Save fit step 0 results
	  hmean_step0->SetBinContent( k+1,muBin, timeResFitPointer->Parameter(1));
	  hsigm_step0->SetBinContent( k+1,muBin, timeResFitPointer->Parameter(2));
	}
	delete hist1d;
      }
    }

    for (int ibin=1; ibin<=ntmp; ibin++) {
      double dtmp=hmeantmp->GetBinContent(ibin);
      double chi2tmp=hchi2tmp->GetBinContent(ibin);
      //      if (!(dtmp>tcmin && dtmp<tcmax) || chi2tmp > 20) {
      if (!(dtmp>tcmin && dtmp<tcmax)) {
        hmeantmp->SetBinContent(ibin,0.);
        hmeantmp->SetBinError(ibin,0.);
      }
    }
    hmnsg[0][k] = new TH1D(*hmeantmp);
    hmnsg[1][k] = new TH1D(*hsigmtmp);
    hmnsg[0][k]->SetName(Form("hmean_%d",imom));
    hmnsg[1][k]->SetName(Form("hsigm_%d",imom));

    for (int idx=0; idx<2; idx++) {
      ptrres[idx]=hmnsg[idx][k]->Fit(PolyFormula(npars-1).Data(),"S");
      int ires=ptrres[idx];
      cout << ptrres[idx]->Ndf() << endl;
      if (ires) {
        cout << "Fit abnormally terminated!!! ires: " << ires << " idx " << idx << " imom " << imom <<  endl;
        exit(-1);
      } else if (ptrres[idx]->Chi2()/ptrres[idx]->Ndf() > 25.){
	cout << "Bad fiT! ires: " << ires << " idx " << idx << " imom " << imom << " chi2/ndf " << (ptrres[idx]->Chi2()/ptrres[idx]->Ndf()) <<  endl;
      }
    }
    
    for (int i=0; i<npars; i++) {
      arpar[0][i][k]=ptrres[0]->Parameter(i);//corrected time mean parameters
      arpar[1][i][k]=ptrres[1]->Parameter(i);//corrected time RMS parameters
      arparerr[0][i][k]=ptrres[0]->ParError(i);//corrected time mean parameters
      arparerr[1][i][k]=ptrres[1]->ParError(i);//corrected time RMS parameters
      //        cout << arpar[0][i][k] << ", " << arpar[1][i][k] << endl;
    }
  }
//  exit(-1);

  // Save fit step 1 results
  for (int k = 0; k < nmom; k++){
    for (int muBin = 1; muBin < ntmp; muBin++){
      double testMean = 0.;
      double testSigma = 0.;
      double mupow = 1.;
      for (int i = 0; i < npars; i++){
	testMean += arpar[0][i][k]*mupow;
	testSigma += arpar[1][i][k]*mupow;
	mupow *= hmeantmp->GetXaxis()->GetBinLowEdge(muBin);
      }
      hmean_step1->SetBinContent( k+1,muBin, testMean);
      hsigm_step1->SetBinContent( k+1,muBin, testSigma);
    }
  }
  
  TFile *fout = new TFile(Form("%d_tpdfpar.root",PID),"RECREATE");
  
  TH2D *htpdfpar[2];
  htpdfpar[0] = new TH2D("htpdfparmn","Time pdf mean fit parameters",npars,0,npars,nthrdpars,0,nthrdpars);
  htpdfpar[1] = new TH2D("htpdfparsg","Time pdf sigma fit parameters",npars,0,npars,nthrdpars,0,nthrdpars);
  
  if (!(momfitrang[0]>armom[0])) momfitrang[0]=armom[0];
  if (!(momfitrang[1]<armom[nmom-1])) momfitrang[1]=armom[nmom-1];
  
  for (int i=0; i<npars; i++) {
    cout << "i  " << i << endl;
    for (int idx=0; idx<2; idx++) {
      if (fNoErrorbars) {
        gtpar[idx][i] = new TGraphErrors(nmom,armom,arpar[idx][i],NULL,NULL);
      }
      else {
        gtpar[idx][i] = new TGraphErrors(nmom,armom,arpar[idx][i],NULL,arparerr[idx][i]);
      }
      gtpar[idx][i]->SetName(Form("gtc%spar_%d",strParNam[idx].Data(),i));
      
      int nthrdtmp=nthrdpars+1;
      int ires;
      do {
        nthrdtmp--;
        
        cout << "nthrd=" << nthrdtmp << endl;
        if (nthrdtmp<6) {
          cout << "Fit abnormally terminated!!!" << endl;
          exit(-1);
        }
        
        ptrprres[idx]=gtpar[idx][i]->Fit(PolyFormula(nthrdtmp-1).Data(),"S","",momfitrang[0],momfitrang[1]);
        ires=ptrprres[idx];
        
      } while (ires);
      
      for (int j=0; j<nthrdpars; j++) {
        if (j<nthrdtmp) {
          thdpr[idx][i][j]=ptrprres[idx]->Parameter(j);
        }
        else {
          thdpr[idx][i][j]=0.;
        }
        
        htpdfpar[idx]->SetBinContent(i+1,j+1,thdpr[idx][i][j]);
      }
      
      gtpar[idx][i]->SetMarkerStyle(2);
      gtpar[idx][i]->Write();
      delete gtpar[idx][i];
    }
    
  }
  
  for (int j=0; j<nmom; j++) {
    hmnsg[0][j]->Write();
    hmnsg[1][j]->Write();
    delete hmnsg[0][j];
    delete hmnsg[1][j];
  }
  
  TH1D *htpdfinfo = new TH1D("htpdfinfo","",10,0,10);
  htpdfinfo->SetBinContent(1,npars);
  htpdfinfo->SetBinContent(2,h3d->GetYaxis()->GetBinCenter(1));
  htpdfinfo->SetBinContent(3,h3d->GetYaxis()->GetBinCenter(h3d->GetNbinsY()));
  htpdfinfo->SetBinContent(4,nthrdpars);
  htpdfinfo->SetBinContent(5,momfitrang[0]);
  htpdfinfo->SetBinContent(6,momfitrang[1]);
  htpdfinfo->SetBinContent(7,pofst);
  
  htpdfinfo->Write();
  delete htpdfinfo;

  hmean_step0->Write();
  hsigm_step0->Write();
  hmean_step1->Write();
  hsigm_step1->Write();
  
  htpdfpar[0]->Write();
  htpdfpar[1]->Write();
  delete htpdfpar[0];
  delete htpdfpar[1];
  
  delete fout;
  
  delete fitmp;

  return 0;
}
