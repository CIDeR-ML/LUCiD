//script to normalize the cherenkov profiles and put them into one file

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <TH1D.h>
#include <TH2D.h>
#include <TH3F.h>
#include <TH3D.h>
#include <TGraphErrors.h>
#include <TFile.h>

using namespace std;

const int nNphotpars=10;
const int nsthrpars=7;
const int nIisopars=12;

const int nsect=3;
const int npars=9;
const int nparnum=2+(npars-2)*nsect+nsect+1;

Double_t xbounds[nsect+1]={2.,5.5,7.5,10.};

void ShiftPoly(int nord, double mu, double *Csft){//Get polynomial coefficients expanded at mu
  double *Corg = new double[nord+1];
  for (int j=0; j<=nord; j++) Corg[j]=Csft[j];
  for (int j=0; j<=nord; j++) {
    Csft[j]=0.;
    for (int i=j; i<=nord; i++) {
      Csft[j]+=Corg[i]*TMath::Binomial(i,j)*pow(mu,i-j);
    }
  }
  delete[] Corg;
}

void fconpoly(Double_t xval, Double_t *par, Double_t& tmpv, Double_t& tmpd) {
  const int npar=npars;
  int ibase=2+(npar-2)*nsect;//index of the first boundary
  double params[npar];
  double xtmp,xpow,tmpval=0.,tmpder=0.;
  bool fBreak=false;
  for (int isect=0; isect<nsect; isect++) {
    if (xval<par[ibase+isect+1] || isect==nsect-1) {
      xtmp=xval;
      fBreak=true;
    }
    else {
      xtmp=par[ibase+isect+1];
    }
    xtmp-=par[ibase+isect];
    if (isect==0) {
      params[0]=par[0];
      params[1]=par[1];
    }
    else {
      params[0]=tmpval;
      params[1]=tmpder;
    }
    for (int ipar=2; ipar<npar; ipar++) {
      params[ipar]=par[(npar-2)*isect+ipar];
    }
    
    xpow=1.;
    tmpval=0.;
    tmpder=0.;
    for (int ipar=0; ipar<npar; ipar++) {
      tmpval+=params[ipar]*xpow;
      if (ipar<npar-1) tmpder+=(ipar+1.)*params[ipar+1]*xpow;
      xpow*=xtmp;
    }
    if (fBreak) break;
    
  }
  tmpv=tmpval;
  tmpd=tmpder;
  
}

Double_t fconpolywrap(Double_t *x, Double_t *par) {
  Double_t xval=x[0];
  Double_t tmpval,tmpder;
  fconpoly(xval,par,tmpval,tmpder);
  return tmpval;
}

//TString PolyFormula(int nord) {
//  TString strPoly="1++x";
//  for (int i=2; i<=nord; i++) {
//    strPoly+=Form("++x^%d",i);
//  }
//  
//  return strPoly;
//}

void LogtheGraph(TGraph *gtmp, double ofst, bool fYaxis=false) {
  Double_t *xarr=gtmp->GetX();
  if (fYaxis) xarr=gtmp->GetY();
  int npts=gtmp->GetN();
  for (int ipt=0; ipt<npts; ipt++) {
    xarr[ipt]=log(xarr[ipt]-ofst);
  }
}

double FitSect(TGraph *gtmp, TF1 **func, int isect, int ipeak, int flgFirst=0) {
  
  int irltv2peak=isect-ipeak;
  
  double xorig;
  if (irltv2peak<0) {
    xorig=xbounds[isect+1];
  }
  else {
    xorig=xbounds[isect];
  }
  
  TGraph *gofst = new TGraph(*gtmp);
  gofst->SetName(Form("%s_%d",gofst->GetName(),isect));
  Double_t *xarr=gofst->GetX();
  int npts=gofst->GetN();
  int ndf=0;
  for (int ipt=0; ipt<npts; ipt++) {
    if (xarr[ipt]>xbounds[isect] && xarr[ipt]<xbounds[isect+1]) ndf++;
    xarr[ipt]-=xorig;//shift the connection point to 0
  }
  
  double wlen=xbounds[isect+1]-xbounds[isect];
  if (flgFirst) {
    double wtmp=npars*wlen/ndf;
    if (wlen>2.*wtmp) wlen=2.*wtmp;
    if (!(ndf>npars)) return -1.;
  }
  
  double xrangL,xrangR;
  if (irltv2peak>0) {
    xrangL=0.;
    xrangR=wlen;
    if (flgFirst==0) {
      xrangR*=1.2;
    }
  }
  else if (irltv2peak<0) {
    xrangL=-wlen;
    xrangR=0.;
    if (flgFirst==0 && isect!=0) {
      xrangL*=1.2;
    }
  }
  else {
    xrangL=-wlen*0.2;
    xrangR=wlen*1.2;
  }
  
  func[isect] = new TF1(Form("func_%s",gofst->GetName()),Form("pol%d",npars-1),xrangL,xrangR);
  func[isect]->SetLineColor(2);
  
  if (flgFirst==0) {
    if (irltv2peak>0) {
      //    cout << xbounds[isect-1] << ", " << xbounds[isect] << endl;
      func[isect]->FixParameter(0,func[isect-1]->Eval(xbounds[isect]-xbounds[isect-1]));
      func[isect]->FixParameter(1,func[isect-1]->Derivative(xbounds[isect]-xbounds[isect-1]));
    }
    else if (irltv2peak<0) {
      double xeval;
      if (irltv2peak==-1) {//next to peak bin
        xeval=0.;
      }
      else {
        xeval=-(xbounds[isect+2]-xbounds[isect+1]);
      }
      func[isect]->FixParameter(0,func[isect+1]->Eval(xeval));
      func[isect]->FixParameter(1,func[isect+1]->Derivative(xeval));
    }
  }
  
  gofst->Fit(func[isect],"","",xrangL,xrangR);//set the margin!!
  
  double tmp = gofst->Chisquare(func[isect])/ndf;
  
//  gofst->Write();
  delete gofst;
  
  return tmp;
}

int FitConPoly(TGraph *gtmp) {
  cout << "Fitting " << gtmp->GetName() << endl;
  
  int ipeak=1;
  
  TF1 *func[nsect];
  FitSect(gtmp,func,1,ipeak);
  FitSect(gtmp,func,0,ipeak);
  FitSect(gtmp,func,2,ipeak);
  
  double pararr[nsect][npars];
  for (int isect=0; isect<nsect; isect++) {
    for (int ipar=0; ipar<npars; ipar++) {
      pararr[isect][ipar]=func[isect]->GetParameter(ipar);
    }
    if (isect<ipeak) ShiftPoly(npars-1,xbounds[isect]-xbounds[isect+1],pararr[isect]);
    delete func[isect];
  }
  
  TF1 *fconn = new TF1(Form("func_%s",gtmp->GetName()),fconpolywrap,xbounds[0],xbounds[nsect],nparnum);
  fconn->SetLineColor(2);
  //  fconn->SetNpx(1000);
  for (int i=0; i<=nsect; i++) {
    fconn->FixParameter(2+(npars-2)*nsect+i,xbounds[i]);//set boundary positions
  }
  fconn->SetParameter(0,pararr[0][0]);
  fconn->SetParameter(1,pararr[0][1]);
  for (int isect=0; isect<nsect; isect++) {
    for (int ipar=2; ipar<npars; ipar++) {
      fconn->SetParameter((npars-2)*isect+ipar,pararr[isect][ipar]);
    }
  }
  
  
//  for (int i=1; i<=2; i++) {
//    fconn->ReleaseParameter(2+(npars-2)*nsect+i);//set boundary positions
//  }
  
//  for (int i=0; i<2+(npars-2)*nsect; i++) fconn->FixParameter(i,fconn->GetParameter(i));
  
  gtmp->Fit(fconn,"","",xbounds[0],xbounds[nsect]);
  
  Double_t tmppars[100];
  for (int ipar=0; ipar<nparnum; ipar++) {
    tmppars[ipar]=fconn->GetParameter(ipar);
  }
  
  int nparbin=npars*nsect+nsect+1;
  TH1D *hpar = new TH1D(Form("%s_pars",gtmp->GetName()),Form("%s parameters",gtmp->GetName()),nparbin,0.,nparbin);
  
  int ibase=2+(npars-2)*nsect;//index of the first boundary
  int isect=0;
  while (1) {
    double xbedgtmp=fconn->GetParameter(ibase+isect);//low edge of the bin
    hpar->SetBinContent(npars*nsect+isect+1,xbedgtmp);
    if (isect>=nsect) break;
    
    double partmp[npars];
    
    Double_t tmpval,tmpder;
    fconpoly(xbedgtmp,tmppars,tmpval,tmpder);
    
    partmp[0]=tmpval;
    partmp[1]=tmpder;
    
    for (int ipar=2; ipar<npars; ipar++) {
      partmp[ipar]=fconn->GetParameter(ipar+(npars-2)*isect);
    }
    ShiftPoly(npars-1,-xbedgtmp,partmp);//Get coefficients at zero
    
    int iz=isect*npars+1;
    for (int ipar=0; ipar<npars; ipar++) {
      hpar->SetBinContent(iz+ipar,partmp[ipar]);
    }
    isect++;
  }
  
  hpar->Write();
  delete hpar;
  
  return 0;
  
}

int writecprof(int PID){//PDG
  
  TH2D *hI3d_nsect[3];
  TH3D *hI3d_par[3];
  
  cout << "PDG=" << PID << endl;
  
  TFile *fin = new TFile(Form("CProf_%d_fit_out.root",PID));
  if (fin->IsZombie()) exit(-1);
  
  TFile *fin2 = new TFile(Form("CProf_%d.root",PID));
  if (fin2->IsZombie()) exit(-1);
  
  TFile *fout = new TFile(Form("CProf_%d_fit.root",PID),"RECREATE");
  
  TH1D *hprofinf = (TH1D*)fin->Get("hprofinf");
  hprofinf->Write();
  
//  int npars=(int)(hprofinf->GetBinContent(1)+0.5);
//  int nsectmax=(int)(hprofinf->GetBinContent(2)+0.5);
  double momofst=hprofinf->GetBinContent(3);
//  momminstp[iPID]=hprofinf->GetBinContent(4);
  xbounds[0]=hprofinf->GetBinContent(5);
  xbounds[nsect]=hprofinf->GetBinContent(6);
  
  delete hprofinf;
  
  TGraph *gNphot = (TGraph*)fin2->Get("gNphot");
  TGraph *gS = (TGraph*)fin2->Get("gsthr");
  
  gNphot->Write();
  gS->Write();
  
  LogtheGraph(gNphot,momofst);
  LogtheGraph(gNphot,0.,true);
  LogtheGraph(gS,momofst);
  LogtheGraph(gS,0.,true);
  
  FitConPoly(gNphot);
  FitConPoly(gS);
  
  for (int i=0; i<2; i++) {
    TH1D *h1dtmp = (TH1D*)fin2->Get(Form("hI_iso_%d",i+1));
    
    int nmomtmp=h1dtmp->GetNbinsX();
    TGraph *gIniso = new TGraph(nmomtmp);
    gIniso->SetName(Form("gI_iso_%d",i+1));
    for (int j=0; j<nmomtmp; j++) {
      gIniso->SetPoint(j,h1dtmp->GetXaxis()->GetBinLowEdge(j+1),h1dtmp->GetBinContent(j+1));
    }
    
    gIniso->Write();
    h1dtmp->Write();
    
    LogtheGraph(gIniso,momofst);
    LogtheGraph(gIniso,0.,true);
    
    FitConPoly(gIniso);
  }
  
  for (int idx=0; idx<3; idx++) {
    cout << "Writing idx=" << idx << endl;
    hI3d_nsect[idx] = new TH2D(*(TH2D*)fin->Get(Form("hI3d_nsect_%d",idx)));
    hI3d_par[idx] = new TH3D(*(TH3D*)fin->Get(Form("hI3d_par_%d",idx)));
    hI3d_nsect[idx]->Write();
    hI3d_par[idx]->Write();
    delete hI3d_nsect[idx];
    delete hI3d_par[idx];
  }
  
  delete fout;
  delete fin2;
  delete fin;
}
