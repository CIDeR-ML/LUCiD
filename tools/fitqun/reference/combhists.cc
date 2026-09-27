//script to normalize the cherenkov profiles and put them into one file

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <TH1D.h>
#include <TH2D.h>
#include <TH3D.h>
#include <TGraphErrors.h>
#include <TFile.h>

using namespace std;

int combhists(int PID){
  const int nmommax=200;
  Double_t armom[nmommax],*arXbins,*arYbins;
  TH2D *hist2d[nmommax];
  
  int nmom=0;
  for (int imom=1; imom<=11000; imom++) {
    TFile *fitmp = new TFile(Form("%d_%d_hist_sum.root",PID,imom));
    if (!fitmp->IsZombie()) {
      cout << imom << endl;
      gROOT->cd();
      hist2d[nmom] = new TH2D(*(TH2D*)fitmp->Get("htimepdf"));
      armom[nmom]=imom;
      
      nmom+=1;
    }
    delete fitmp;
    if (nmom>=nmommax) exit(-1);
  }
  armom[nmom]=armom[nmom-1]+1.;
  
  int nXbins=hist2d[0]->GetNbinsX();
  int nYbins=hist2d[0]->GetNbinsY();
  arXbins = new Double_t[nXbins+1];
  for (int i=0; i<=nXbins; i++) arXbins[i]=hist2d[0]->GetXaxis()->GetBinLowEdge(i+1);
  arYbins = new Double_t[nYbins+1];
  for (int i=0; i<=nYbins; i++) arYbins[i]=hist2d[0]->GetYaxis()->GetBinLowEdge(i+1);
  
  TFile *fout = new TFile(Form("%d_tpdfhist.root",PID),"RECREATE");
  
  TH3D *h3d = new TH3D("hist_tpdf","",nXbins,arXbins,nYbins,arYbins,nmom,armom);
  for (int k=1; k<=nmom; k++) {
    for (int i=1; i<=nXbins; i++) {
      for (int j=1; j<=nYbins; j++) {
        h3d->SetBinContent(i,j,k,hist2d[k-1]->GetBinContent(i,j));
      }
    }
  }
  h3d->Sumw2();
  h3d->Write();
  
  delete fout;
  
  return 0;
}
