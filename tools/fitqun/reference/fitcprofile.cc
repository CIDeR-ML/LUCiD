//script to normalize the cherenkov profiles and put them into one file

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <TMath.h>
#include <TF1.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TH3F.h>
#include <TH3D.h>
#include <TGraphErrors.h>
#include <TFile.h>

#include <time.h>

using namespace std;

const int npars=5;//n+1

int nsectmax=50;
int nparnum=2+(npars-2)*nsectmax+nsectmax+1;
int nparbin=npars*nsectmax+nsectmax+1;

int nsect;

int fQuiet=1;
int fFullFit=1;
TString strFitflg[2]={"","Q"};

double xbounds[201];//has to be larger than nsectmax
double xofst=0.;
//double xminstp=0.1;
double xminstp=0.2;
//const int nstprat=6;
//const double xbigstp=nstprat*xminstp;

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

double FitSect(TGraphErrors *gtmp, TF1 **func, int isect, int ipeak, int flgFirst=0) {
  
  int irltv2peak=isect-ipeak;
  
  double xorig;
  if (irltv2peak<0) {
    xorig=xbounds[isect+1];
  }
  else {
    xorig=xbounds[isect];
  }
  
  TGraphErrors *gofst = new TGraphErrors(*gtmp);
  gofst->SetName(Form("%s_%d",gofst->GetName(),nsect));
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
    if (!(ndf>npars)) {
      delete gofst;
      return -1.;
    }
//    if (wlen<1.2*wtmp) wlen=1.2*wtmp;
//    if (!fQuiet) cout << "wlen=" << wlen << endl;
  }
  
  double xrangL,xrangR;
  if (irltv2peak>0) {
    xrangL=0.;
    xrangR=wlen;
    if (flgFirst==0) {
      xrangR*=1.3;
    }
//    else {
////      xrangL=-0.2*xrangR;
//      xrangR*=1.3;
//    }
  }
  else if (irltv2peak<0) {
    xrangL=-wlen;
    xrangR=0.;
    if (flgFirst==0) {
      xrangL*=1.3;
    }
//    else {
////      xrangR=-0.2*xrangL;
//      xrangL*=1.3;
//    }
  }
  else {
    xrangL=-(xbounds[isect+1]-xbounds[isect])*0.3;
    xrangR=(xbounds[isect+1]-xbounds[isect])*1.3;
  }
  
  func[isect] = new TF1(Form("func_%s",gofst->GetName()),Form("pol%d",npars-1),xrangL,xrangR);
  
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
  
  gofst->Fit(func[isect],"Q","",xrangL,xrangR);//set the margin!!
  
  double tmp = gofst->Chisquare(func[isect])/ndf;
  
//  gofst->Write();
  delete gofst;
  
  return tmp;
}

int ExpndBin(TGraphErrors *gtmp, TF1 **func, int isect, int ipeak, double chsqthr=1., double chsqrat=4., bool fPreset=false) {
  
  int irltv2peak=isect-ipeak;
  
  if (!fPreset) {
    if (irltv2peak<0) {
      xbounds[isect]=xbounds[isect+1]-xminstp;
    }
    else {
      xbounds[isect+1]=xbounds[isect]+xminstp;
    }
  }
  
  int nmom=gtmp->GetN();
  Double_t *arX=gtmp->GetX();
  Double_t *arY=gtmp->GetY();
  
  func[isect]=NULL;
  TF1 *fold=NULL;
  int iret=0;
  int iFit=0;
  int ipts;//# of points in the window
  double chsq=0.,chsqold,chsq1st=1.;
  int fDir=1;
  int iLoop=0;
  while (1) {
    double yL=-1,yR=-1;
    ipts=0;
    for (int i=0; i<nmom; i++) {
      
      if (arX[i]>xbounds[isect]) {
        if (yL<0.) yL=arY[i];
        if (arX[i]>xbounds[isect+1]) {
          yR=arY[i-1];
          break;
        }
        ipts++;
      }
    }
    
    if (ipts>npars) {
//    if (ipts>3) {
      fold=func[isect];
//      chsqold=chsq;
      
      double xletmp = xbounds[isect];
      double xhetmp = xbounds[isect+1];
      
      if (iFit) {
        if (fDir==-1) {
          xbounds[isect+1]=xbounds[isect]+xminstp;
        }
        else {
          xbounds[isect]=xbounds[isect+1]-xminstp;
        }
      }
      else {
        iLoop=0;
      }
      double dret=FitSect(gtmp,func,isect,ipeak,1);
      if (dret>0.) {
        chsq1st = (chsq1st*iLoop+dret)/(iLoop+1);
        delete func[isect];
      }
      
      xbounds[isect]=xletmp;
      xbounds[isect+1]=xhetmp;
      
//      if (iFit==0) {
//        chsq1st = FitSect(gtmp,func,isect,ipeak,1);
//        delete func[isect];
//      }
      chsq = FitSect(gtmp,func,isect,ipeak);
      if (!fQuiet) cout << isect << ", " << ipeak << ", " << xbounds[isect+1]-xbounds[isect] << ", " << chsq << ", " << chsq1st << ", " << dret << endl;
//      cout << fold << ", " << func[isect] << ", " <<  chsq << endl;
      if (chsq>chsq1st*chsqrat && chsq>chsqthr) {//evaluate chisq at next section only and use that?
//      if (chsq>chsqold*1.2 && chsq>1.0) {
        if (iFit!=0) {
          if (fDir==-1) {
            xbounds[isect]+=xminstp;
          }
          else {
            xbounds[isect+1]-=xminstp;
          }
          delete func[isect];
          func[isect]=fold;
          
//          delete func[isect];
//          chsq = FitSect(gtmp,func,isect,ipeak);
          if (!fQuiet) cout << xbounds[isect] << ", " << xbounds[isect+1] << ", " << func[isect]->GetParameter(0) << ", " << func[isect]->GetParameter(1) << endl;
          
          break;
        }
      }
      iFit=1;
      if (fold) delete fold;
    }
    
    if (irltv2peak>0) {
      fDir=1;
    }
    else if (irltv2peak<0) {
      fDir=-1;
    }
    else {//peak
      if (yL>yR) fDir=-1;
      else fDir=1;
      
      if (xbounds[isect]<arX[0]) fDir=1;
      if (xbounds[isect+1]>arX[nmom-1]) fDir=-1;
    }
    if (fDir==-1) {
      if (xbounds[isect]<arX[0]) {
        iret=-1;
        break;
      }
      xbounds[isect]-=xminstp;
    }
    else {
      if (xbounds[isect+1]>arX[nmom-1]) {
        iret=1;
        break;
      }
      xbounds[isect+1]+=xminstp;
    }
    
    iLoop++;
    
  }
  if (!(ipts>npars)) iret*=10;
  
  return iret;
}

int FitIt(TGraphErrors *gtmp, TF1*& fconn) {
  
  int ireturn=0;
//  int imaxpos=0;
  double maxpos=0.,maxval=-1.;
  int nmom=gtmp->GetN();
  Double_t *arX=gtmp->GetX();
  Double_t *arY=gtmp->GetY();
//  for (int imom=0; imom<nmom; imom++) {
//    if (arY[imom]>maxval) {
//      maxval=arY[imom];
//      maxpos=arX[imom];//peak position
////      imaxpos=imom;
//    }
//  }
  
  maxpos=arX[nmom-1];
  
  double xtmp=arX[0]-1e-3;
  while (1) {
    xtmp+=xminstp;
    if (maxpos<xtmp) break;
  }
  xtmp-=xminstp;//lowedge of the peak bin
  
  TF1 *func[201];
  int ipeak;
  
  double chsqthresh=1.;
  double chsqthrrat=8.;
  
  while (1) {
    int iItr=0;
    if (!fQuiet) cout << "chsqthr=" << chsqthresh << endl;
    
    nsect=1;
    ipeak=0;
    xbounds[0]=xtmp;
    
    int iret=ExpndBin(gtmp,func,ipeak,ipeak,chsqthresh,chsqthrrat);
    
    while (!(iret<0)) {//add sections to the left
      if (nsect>=nsectmax) {
        iItr=1;
        iret=0;
        break;
      }
      
      for (int idx=ipeak+1; idx>=0; idx--) {
        func[idx+1]=func[idx];
        xbounds[idx+1]=xbounds[idx];
      }
      nsect++;
      ipeak++;
      iret=ExpndBin(gtmp,func,0,ipeak,chsqthresh,chsqthrrat);
    }
    if (iret<-1) {//too few data points in the new bin, merge it
      nsect--;
      ipeak--;
//      xbounds[0]=xbounds[1];
      if (func[0]) delete func[0];
      for (int idx=0; idx<nsect; idx++) {
        func[idx]=func[idx+1];
        xbounds[idx+1]=xbounds[idx+2];
      }
    }
    
    xbounds[0]=arX[0]-1e-3;
    delete func[0];
    iret=ExpndBin(gtmp,func,0,ipeak,chsqthresh,chsqthrrat,true);
    
//    while (!(iret>0)) {
//      if (nsect>=nsectmax) {
//        iItr=1;
//        iret=0;
//        break;
//      }
//      
//      nsect++;
//      iret=ExpndBin(gtmp,func,nsect-1,ipeak,chsqthresh,chsqthrrat);
//    }
//    if (iret>1) {//too few data points in the new bin, merge it
//      nsect--;
//      //    xbounds[nsect]=xbounds[nsect+1];
//    }
    
//    if (nsect<2) iItr=-1;
    
    if (iItr>0) {
      if (!fQuiet) cout << "nsect exceeded maximum value!" << endl;
      if (chsqthresh>5.) {
        ireturn=-1;
        break;
      }
      chsqthresh+=1.;
    }
    else if (iItr<0) {
      if (!fQuiet) cout << "nsect too small!" << endl;
      chsqthresh*=0.5;
      chsqthrrat*=0.7;
      if (chsqthresh<0.1) {
        ireturn=1;
        break;
      }
    }
    else {
      break;
    }
    
    for (int isect=0; isect<nsect; isect++) {
      delete func[isect];
    }
  }
  
  if (!fQuiet) cout << "nsect=" << nsect << endl;
  
  double pararr[201][npars];
  for (int isect=0; isect<nsect; isect++) {
    for (int ipar=0; ipar<npars; ipar++) {
      pararr[isect][ipar]=func[isect]->GetParameter(ipar);
    }
    if (isect<ipeak) ShiftPoly(npars-1,xbounds[isect]-xbounds[isect+1],pararr[isect]);
    delete func[isect];
  }
  
  fconn = new TF1(Form("func_%s",gtmp->GetName()),fconpolywrap,xbounds[0],xbounds[nsect],nparnum);
//  fconn->SetNpx(1000);
  for (int i=0; i<=nsect; i++) {
    fconn->FixParameter(2+(npars-2)*nsect+i,xbounds[i]);//set boundary positions
  }
  for (int i=2+(npars-2)*nsect+nsect+1; i<nparnum; i++) {
    fconn->FixParameter(i,-1.);//unused parameters
  }
  fconn->SetParameter(0,pararr[0][0]);
  fconn->SetParameter(1,pararr[0][1]);
  for (int isect=0; isect<nsect; isect++) {
    for (int ipar=2; ipar<npars; ipar++) {
      fconn->SetParameter((npars-2)*isect+ipar,pararr[isect][ipar]);
    }
  }
  
  if (fFullFit==0) for (int i=0; i<2+(npars-2)*nsect; i++) fconn->FixParameter(i,fconn->GetParameter(i));
  
  gtmp->Fit(fconn,strFitflg[fQuiet].Data(),"",xbounds[0],xbounds[nsect]);
  
  return ireturn;
}

int main(int argc, char* argv[]){//PDG
  
  int PID=atoi(argv[1]);
  if (PID<0) {
    fQuiet=0;
    PID=-PID;
  }
  if (PID>10000) {
    fFullFit=0;//prefit only
    PID-=10000;
  }
  cout << "Particle Code: " << PID << endl;
  
  int iR0bin=-1;
  if (argc>2) {
    iR0bin=atoi(argv[2]);
  }
  int ith0bin=-1;
  if (argc>3) {
    ith0bin=atoi(argv[3]);
  }
  
  if (argc>4) {
    xminstp=atof(argv[4]);
  }
  cout << "xminstp=" << xminstp << endl;
  
  double mommin=0.;//ignore momentum below this value
  if (PID==11) {
    xofst=-20.;
    mommin=2.5;
  }
  else if (PID==13) {
    xofst=100.;
  }
  else if (PID==211) {
    xofst=140.;
  }
  else if (PID==321) {
    xofst=530.;
//    xminstp=0.06;
  }
  else if (PID==2212) {
    xofst=1020.;
//    xminstp=0.1;
//    xminstp=0.06;
  }
  else {
    cout << "Unknown particle type!" << endl;
    exit(-1);
  }
  
  TFile *fin,*fout;
  const int nmommax=2000;
  Double_t armom[nmommax],arval[nmommax],arerr[nmommax];
  TH3F *hI3d[3];
  TGraphErrors *gtmp;
  
  fin = new TFile(Form("CProf_%d.root",PID));
  if (fin->IsZombie()) {
    cout << "Cannot open histogram file!" << endl;
    exit(-1);
  }
  
  for (int i=0; i<3; i++) {
    hI3d[i] = (TH3F*)fin->Get(Form("hI3d_%d",i));
  }
  fout = new TFile(Form("CProf_%d_fit_out.root",PID),"RECREATE");
  
  cout << "Fitting..." << endl;
  int nR0bin = hI3d[0]->GetNbinsX();
  int nth0bin = hI3d[0]->GetNbinsY();
  int nmom = hI3d[0]->GetNbinsZ();
  
  int imomofst=0;
  for (int imom=0; imom<nmom; imom++) {
    armom[imom-imomofst]=hI3d[0]->GetZaxis()->GetBinLowEdge(imom+1);
    if (armom[imom-imomofst]<mommin) {
      imomofst++;
      continue;
    }
    armom[imom-imomofst]=log(armom[imom-imomofst]-xofst);
  }
  nmom-=imomofst;
  cout << Form("Skipping first %d points.",imomofst) << endl;
  
  Double_t *R0binEdgs = new Double_t[nR0bin+1];
  Double_t *th0binEdgs = new Double_t[nth0bin+1];
  Double_t *parbinEdgs = new Double_t[nparbin+1];
  
  for (int i=0; i<=nR0bin; i++) R0binEdgs[i]=hI3d[0]->GetXaxis()->GetBinLowEdge(i+1);
  for (int i=0; i<=nth0bin; i++) th0binEdgs[i]=hI3d[0]->GetYaxis()->GetBinLowEdge(i+1);
  for (int i=0; i<=nparbin; i++) parbinEdgs[i]=i-0.5;
  
  time_t timer = time(NULL);
  
  int novfl=0;
  TH1D *hnsect = new TH1D("hnsect","Number of sections",nsectmax,0.5,nsectmax+0.5);
  
  Double_t xminmax[2];
  TH3D *hI3d_par[3];
  TH2D *hI3d_par_chi2[3];
  TH2D *hI3d_nsect[3];
  for (int i=0; i<3; i++) {
    hI3d_par[i] = new TH3D(Form("hI3d_par_%d",i),Form("I_{%d} parameters",i),nR0bin,R0binEdgs,nth0bin,th0binEdgs,nparbin,parbinEdgs);
    hI3d_par_chi2[i] = new TH2D(Form("hI3d_par_%d_chi2",i),Form("I_{%d} parameters",i),nR0bin,R0binEdgs,nth0bin,th0binEdgs);
    hI3d_nsect[i] = new TH2D(Form("hI3d_nsect_%d",i),Form("I_{%d} nsect",i),nR0bin,R0binEdgs,nth0bin,th0binEdgs);
    cout << "n=" << i << endl;
    for (int j=1; j<=nR0bin; j++) {
      if (iR0bin>0 && j!=iR0bin) continue;
      cout << "iR0=" << j << endl;
      for (int k=1; k<=nth0bin; k++) {
        if (ith0bin>0 && k!=ith0bin) continue;
        for (int imom=0; imom<nmom; imom++) {
          arval[imom]=hI3d[i]->GetBinContent(j,k,imom+imomofst+1);
//          arerr[imom]=sqrt(arval[imom])/10.;
          arerr[imom]=arval[imom]/100.;
        }
        
        gtmp = new TGraphErrors(nmom,armom,arval,NULL,arerr);
        gtmp->SetName(Form("I_%d_%d_%d",i,j,k));
        TF1 *fconn = NULL;
        int iret = FitIt(gtmp,fconn);
        
        if (iret==-1) {
          novfl++;
          cout << "Overflow! ith0=" << k << endl;
        }
        else if (iret==1) {
          cout << "nsect=1! ith0=" << k << endl;
        }
        
        Double_t tmppars[1000];
        for (int ipar=0; ipar<nparnum; ipar++) {
          tmppars[ipar]=fconn->GetParameter(ipar);
        }
        
        int ibase=2+(npars-2)*nsect;//index of the first boundary
        int isect=0;
        while (1) {
          double xbedgtmp=fconn->GetParameter(ibase+isect);//low edge of the bin
          hI3d_par[i]->SetBinContent(j,k,npars*nsect+isect+1,xbedgtmp);
          if (isect>=nsect) break;
          
          double partmp[npars];
          
          Double_t tmpval,tmpder;
          fconpoly(xbedgtmp,tmppars,tmpval,tmpder);
          
          partmp[0]=tmpval;
          partmp[1]=tmpder;
//          partmp[0]=fconn->Eval(xbedgtmp);
//          partmp[1]=fconn->Derivative(xbedgtmp);
          
          for (int ipar=2; ipar<npars; ipar++) {
            partmp[ipar]=fconn->GetParameter(ipar+(npars-2)*isect);
          }
          ShiftPoly(npars-1,-xbedgtmp,partmp);//Get coefficients at zero
          
          int iz=isect*npars+1;
          for (int ipar=0; ipar<npars; ipar++) {
            hI3d_par[i]->SetBinContent(j,k,iz+ipar,partmp[ipar]);
          }
          isect++;
        }
        for (int ipar=(npars+1)*nsect+1; ipar<nparbin; ipar++) {
          hI3d_par[i]->SetBinContent(j,k,ipar+1,-9999.);
        }
        xminmax[0]=tmppars[ibase];
        xminmax[1]=tmppars[ibase+nsect];
        
        int ndf=nmom;
        hI3d_par_chi2[i]->SetBinContent(j,k,gtmp->Chisquare(fconn)/ndf);
        hI3d_nsect[i]->SetBinContent(j,k,nsect);
        
        gtmp->Write();
        
        if (fconn) delete fconn;
        delete gtmp;
        
        hnsect->Fill(nsect);
      }
    }
    hI3d_par[i]->Write();
    hI3d_par_chi2[i]->SetOption("COLZ");
    hI3d_par_chi2[i]->Write();
    hI3d_nsect[i]->SetOption("COLZ");
    hI3d_nsect[i]->Write();
    
    delete hI3d_par[i];
    delete hI3d_par_chi2[i];
    delete hI3d_nsect[i];
  }
  
  hnsect->Write();
  delete hnsect;
  
  TH1D *hprofinf = new TH1D("hprofinf","Cherenkov profile binning information",10,-0.5,9.5);
  hprofinf->SetBinContent(1,npars+1e-6);
  hprofinf->SetBinContent(2,nsectmax+1e-6);
  hprofinf->SetBinContent(3,xofst);
  hprofinf->SetBinContent(4,xminstp);
  hprofinf->SetBinContent(5,xminmax[0]);
  hprofinf->SetBinContent(6,xminmax[1]);
  hprofinf->Write();
  delete hprofinf;
  
  for (int i=0; i<3; i++) delete hI3d[i];
  
  delete[] R0binEdgs;
  delete[] th0binEdgs;
  delete[] parbinEdgs;
  
  delete fout;
  delete fin;
  
  cout << "Number of overflows: " << novfl << endl;
  std::cout << "Total time elapsed: " << time(NULL)-timer << "s" << std::endl;
}
