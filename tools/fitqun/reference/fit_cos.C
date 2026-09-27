//#include "TROOT.h"
//#include "TString.h"
//#include "TFile.h"
//#include "TH1D.h"
//#include "TPolyFunc.h"
//#include "TCanvas.h"
//#include "TF1.h"
//
//void testpolyfunc2(std::string inFile, std::string geoName) {
void fit_cos(std::string inFile, std::string geoName) {
//{
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  
  gROOT->ProcessLine(".L TPolyFunc.cxx+");
  gStyle->SetStatY(0.75);
  gStyle->SetStatX(0.5);
  gStyle->SetStatH(0.13);

  TCanvas* tcan;
  TString ordernames[10] = {"pol0","pol1","pol2","pol3","pol4","pol5","pol6","pol7","pol8","pol9",};

  double xmin = 0.;
//  double xmin = 0.01;
  double xmax = 1.;

  //  const int npolys=6;
  //  int myorders[10] = {9,3,5,2,1,4,-1,-1,-1,-1};
//  const int npolys=3;
//  int myorders[10] = {1,4,1,-1,-1,-1,-1,-1,-1,-1};
  const int npolys=2;
  int myorders[10] = {2,3,-1,-1,-1,-1,-1,-1,-1,-1};
  //  double conpts[11] = {xmin,0.29, 0.9,xmax};
  double conpts[11] = {xmin,0.6, xmax};
  //  TFile* tf = new TFile("ang_nuprismwide_d800_100000evts_outfile.root");
  TFile* tf = new TFile(inFile.c_str());
  //  TH1D* myhsta;
  TH1F* myhsta;
  //  tf->GetObject("hsta",myhsta);
  //  tf->GetObject("angRespEndCap_100",myhsta); //NuPRISM
  tf->GetObject("angRespAll_100",myhsta);
  //  myhsta->GetXaxis()->SetTitle("Cosine of Photon Incident Angle");
  myhsta->GetXaxis()->SetTitle("cos#eta");

  TPolyFunc* tpf = new TPolyFunc("angResp",xmin,xmax,myorders[0],myorders[1],myorders[2],myorders[3],myorders[4],myorders[5],myorders[6],myorders[7],myorders[8],myorders[9]);
  tpf->SetNpx(10000);
  int itpfpar = 0;
  for (int icon=0; icon<npolys-1; icon++) {
    tpf->SetParameter(itpfpar,conpts[icon+1]);
    itpfpar++;
  }

  for (int ifun=0; ifun<npolys; ifun++) {
    TString funcname = "func";
    funcname += ifun;
    TF1* func = new TF1(funcname,ordernames[myorders[ifun]],xmin,xmax);
    tcan = new TCanvas(funcname,funcname);
    TH1D* thishsta = (TH1D*)myhsta->Clone();
    thishsta->Fit(func,"","",conpts[ifun],conpts[ifun+1]);
    TString filename = "cos_";
    filename += funcname;
    filename += ".eps";
    tcan->Print("filename");
    std::cout << "conpts[" << ifun << "-" << ifun+1 << "] = " << conpts[ifun] << ", " << conpts[ifun+1] << std::endl;
    for (int ipar=0; ipar<func->GetNpar(); ipar++) {
      double mypar = func->GetParameter(ipar);
      if ((ifun==0)||(ipar>1)) {
	tpf->SetParameter(itpfpar,mypar);
	itpfpar++;
      }
    }
  }
  std::cout << "after filling, itpfpar = " << itpfpar << std::endl;

  tcan = new TCanvas("final","final");
  myhsta->Fit(tpf,"","",xmin,xmax);

  tcan->Print((geoName+"_cos_final.png").c_str());
    //  tpf->Draw();
  std::cout << "Functions" << std::endl;
  tpf->PrintSubPolys();
  std::cout << "Derivatives" << std::endl;
  tpf->PrintSubPolys(true);

//  double norm = tpf->GetParameter(npolys-1);
  double norm = tpf->Eval(1.);
  for (int ipar=npolys-1; ipar<itpfpar; ipar++) {
    double oldval = tpf->GetParameter(ipar);
    tpf->SetParameter(ipar,oldval/norm);
  }
  tcan = new TCanvas("scaled","scaled");
  tpf->Draw();
  tcan->Print((geoName+"_cos_final_scaled.png").c_str());
  std::cout << "=== Scaled ===" << std::endl;
  std::cout << "Functions" << std::endl;
  tpf->PrintSubPolys();
  std::cout << "Derivatives" << std::endl;
  tpf->PrintSubPolys(true);

  TF1 * fCopy = new TF1(*tpf);

  TFile* tf2 = new TFile(("angResp_"+geoName+".root").c_str(),"RECREATE");
  //tpf->Write();
  fCopy->Write();
  tf2->Close();

}
