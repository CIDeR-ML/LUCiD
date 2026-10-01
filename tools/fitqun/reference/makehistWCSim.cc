// Compile with 'make makehistWCSim'

#include <iostream>
#include <TFile.h>
#include <TTree.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TGraph.h>
#include <TBranch.h>
#include <TClonesArray.h>
#include <TString.h>

#include <cmath>
#include <cstdlib>

#include "fiTQun.h"
#include "fiTQun_shared.h"
#include "WCSimWrap.h"
#include "TRuntimeParameters.hxx"

#include "WCSimRootEvent.hh"
#include "WCSimRootGeom.hh"


using namespace std;

const double pi=3.1415926535898;
const double c0=29.9792458;//[cm/ns]
double nwtr=1.38;// Water refractive index for TOF subtraction


// Look at track 2. 0th and 1st track are incoming neutrino and target nucleus
// Use zeroth trigger
const int nWCSimTrack = 2;
const int nWCSimTrig = 0;

// Args: WCSim input file, WCSim configuration name, WCSim PMT type, water refractive index, FC flag
int main(int argc, char* argv[]){

  if (argc<3) {
    cout << "Wrong number of arguments. Please provide:\nWCSim file path; fiTQun parameter override file; refractive index for water [optional]; PC flag requirement [optional]" << endl;

    exit(-1);
  }

  if (argc>=4) {//set water refractive index
    nwtr=atof(argv[3]);
  }
  cout << "nwtr=" << nwtr << endl;
  
  // Manually override the PC flag cut
  int fPCflcgut=1;
  if (argc>=5) {
    fPCflcgut=atoi(argv[4]);
  }

  cout << "PCflg cut: " << fPCflcgut << endl;
  
  char fstr[256];
  
  int nifile_name;
  for (int i=0; i<256; i++) {
    fstr[i]=argv[1][i];
    if (argv[1][i]=='\0') {
      nifile_name=i;
      break;
    }
  }
  std::cout << "Input file: " << argv[1] << ";" << nifile_name << std::endl;
  
  for (int i=nifile_name; i>0; i--) {
    if (fstr[i]=='.') {
      fstr[i]='\0';
      break;
    }
  }
  TString ntplFile = fstr;

  // Use WCSimWrap to read WCSim
  TRuntimeParameters::Get().ReadParamOverrideFile(argv[2]);
  WCSimWrap* wc = WCSimWrap::Get( argv[1] ); // This might be redundant given the command below... need SetInput to readGeom
  wc->SetInput( argv[1] );

  wc->LoadEntry(0);  // Again prob redundant
  
  WCSimRootTrigger * trigWCSim = wc->SubEvt(nWCSimTrig);
  WCSimRootTrack   * trackWCSim = (WCSimRootTrack*) trigWCSim->GetTracks()->At(nWCSimTrack);

  int iTrack = 0;
  int ParentID = trackWCSim->GetParenttype();
  while(ParentID!=0){ // Sometimes recoil protons or neutrons come first
    iTrack++;
    trackWCSim = (WCSimRootTrack*) trigWCSim->GetTracks()->At(nWCSimTrack+iTrack);
    ParentID = trackWCSim->GetParenttype();
  }

  WCSimRootGeom    * geomWCSim = wc->Geo();
  
  int iPID;
  int fLoadflg[nPID];
  for (iPID = 0; iPID < nPID; iPID++) fLoadflg[iPID] = 0;
  for (iPID = 0; iPID < nPID; iPID++) {
    if (fiTQun_shared::PIDarr[iPID] == trackWCSim->GetIpnu()) {
      fLoadflg[iPID] = 1;
      break;
    }
  }
  if (iPID >= nPID) {
    std::cout << "Unknown particle type: " << trackWCSim->GetIpnu() << std::endl;
    exit(-1);
  }
  std::cout << "Particle code: " << fiTQun_shared::PIDarr[iPID] << std::endl;
  
  // Get a fiTQun instance
  
  fiTQun * thefit = new fiTQun( wc->NPMT() );
  TRuntimeParameters::Get().ReadParamOverrideFile(argv[2]);

  fiTQun_shared * fqshared=fiTQun_shared::Get(wc->NPMT()); //tyoshida
  int iScat=0;//no scattered light predicted charge
  
  //  fiTQun_shared::Get()->SetScatflg(iScat);
  //  thefit->SetScatflg(iScat); // comented out by tyoshida
  fqshared->SetScatflg(iScat); //tyoshida

  
  thefit->ReadSharedParams(iScat,fLoadflg,true,true,1);

  // Attenuation length and QE normalisation deliberately NOT overridden here.
  // fiTQun_shared's ctor already reads them from the parameter override file
  // (fiTQun_shared.cc:264,291, via the <key><DetName> lookup), so the values the
  // time PDF is built with are the tune's own rather than SK's 6800 cm / 0.1.
  // fiTQun.cc:99 comments out the same overwrite for exactly this reason.

  // QEEff=0.1 and Phi0=1 are a NORMALISATION CONVENTION, not SK physics: the
  // charge axis is filled as log10(Phi0*QEEff*nphot), and fiTQun.cc:466 divides
  // the real Phi0*QEEff*QEEffCorr back out when it evaluates the PDF. Building
  // with the tune's own QEEff (0.02006) while evaluating against 0.1 shifts the
  // axis by log10(0.02006/0.1) = -0.70, i.e. a factor 5 in charge. Both must be
  // set, exactly as the reference does (makehist.cc:120-121).
  fqshared->SetQEEff(0.1);
  fqshared->SetPhi0(-1.,1.);


  // Get true momentum
  double mom = trackWCSim->GetP();
  
  // Not very sure what's happening here? cgs?
  bool fCG=false;       
  if (fiTQun_shared::PIDarr[iPID]==48) {
    fCG=true;
    mom/=10.;
  }

  double Iiso[2],nphot,smax;
  fiTQun_shared::Get()->SetTypemom(iPID,mom,Iiso,nphot,smax);
  
  double smid=smax/2.;
  std::cout << "p=" << mom << "MeV/c, smid: " << smid << std::endl;
  
  TH2D htimepdf("htimepdf","",400,-100,100,125,-2.,3.);
  double qR_R, qR_mu, qR_q, qR_mom, qR_cos;
  // Unbiased charge-vs-distance: sum observed and predicted charge over EVERY
  // PMT in each R bin. Restricting to hit PMTs biases the ratio upward with R,
  // because at large R only upward fluctuations fire while the prediction still
  // counts the PMTs that did not.
  TH1D hsum_q ("hsum_q" ,"sum observed q vs R" ,60,0.,6000.);
  TH1D hsum_mu("hsum_mu","sum predicted mu vs R",60,0.,6000.);
  // Same sums restricted to PMTs the track actually illuminates. Without a cut
  // the ratio is swamped at large R by the dark-noise floor, which every PMT
  // collects and the model correctly does not predict.
  TH1D hsig_q ("hsig_q" ,"sum q, mu>0.5 pe" ,60,0.,6000.);
  TH1D hsig_mu("hsig_mu","sum mu, mu>0.5 pe",60,0.,6000.);
  std::vector<double> qobs;
  TTree *qrtree = new TTree("qR","predicted vs observed charge against distance");
  qrtree->Branch("R",&qR_R); qrtree->Branch("mu",&qR_mu);
  qrtree->Branch("q",&qR_q); qrtree->Branch("mom",&qR_mom); qrtree->Branch("cos",&qR_cos);

  htimepdf.Sumw2();

  // Loop through the events
  for (int nevt = 0; nevt < wc->NEvt(); nevt++){

    cout << "STARTS EVENT " << nevt << endl;
    wc->LoadEntry(nevt);

    trigWCSim = wc->SubEvt(nWCSimTrig);
    trackWCSim = (WCSimRootTrack*) trigWCSim->GetTracks()->At(nWCSimTrack);

    int iTrack = 0;
    ParentID = trackWCSim->GetParenttype();
    while(ParentID!=0){ // Sometimes recoil protons or neutrons come first
      iTrack++;
      trackWCSim = (WCSimRootTrack*) trigWCSim->GetTracks()->At(nWCSimTrack+iTrack);
      ParentID = trackWCSim->GetParenttype();
    }


    double aSubToffs = wc->GetTOffset(nWCSimTrig);

    double TrkParam[7];
    
    // Set vertex
//    TrkParam[0] = trigWCSim->GetVtx(0);
//    TrkParam[1] = trigWCSim->GetVtx(1);
//    TrkParam[2] = trigWCSim->GetVtx(2);
//    // Set time
//    TrkParam[3] = trackWCSim->GetTime();
//    // Set theta
//    TrkParam[4] = acos( trackWCSim->GetDir(2) );
//    // Set phi
//    TrkParam[5] = atan2( trackWCSim->GetDir(1), trackWCSim->GetDir(0) );
//    // Set momentum
//    TrkParam[6] = trackWCSim->GetP();
    
    // NuPRISM 1->2 2->1
    // Set vertex
    
    TVector3 wcSimVtx(trigWCSim->GetVtx(0), trigWCSim->GetVtx(1), trigWCSim->GetVtx(2));
    TVector3 wcSimDir(trackWCSim->GetDir(0), trackWCSim->GetDir(1), trackWCSim->GetDir(2));

    TrkParam[0] = fiTQun_shared::Get()->TfmDetToGlblCoord(wcSimVtx, true, true).X();
    TrkParam[1] = fiTQun_shared::Get()->TfmDetToGlblCoord(wcSimVtx, true, true).Y();
    TrkParam[2] = fiTQun_shared::Get()->TfmDetToGlblCoord(wcSimVtx, true, true).Z();
    //    cout << "Vertex WCSim: " << wcSimVtx.X() << ' ' << wcSimVtx.Y() << ' ' << wcSimVtx.Z() << endl;
    //    cout << "Vertex fQ: " << TrkParam[0] << ' ' << TrkParam[1] << ' ' << TrkParam[2] << endl;
    
    // Set time
    TrkParam[3] = trackWCSim->GetTime();
    // Set theta
    TrkParam[4] = acos( fiTQun_shared::Get()->TfmDetToGlblCoord(wcSimDir, false, true).Z() );
    // Set phi
    TrkParam[5] = atan2( fiTQun_shared::Get()->TfmDetToGlblCoord(wcSimDir, false, true).Y(), 
			 fiTQun_shared::Get()->TfmDetToGlblCoord(wcSimDir, false, true).X() );

    // Set momentum
    TrkParam[6] = trackWCSim->GetP();
    
    // Again ...
    if (fCG) TrkParam[6]/=10.;
  
    //    cout << "Sets track parameters" << endl;
    
    // Calculate the midpoint of the track
    double midpos[3];
    for (int j = 0; j < 3; j++) {
      midpos[j] = TrkParam[j] + smid*fiTQun_shared::Get()->TfmDetToGlblCoord(wcSimDir, false, true)[j];
    }
    

    // Get the predicted charge at each pmt and the PC flag
    double muarr[nPMT_max];
    int PCflg = thefit->Get1Rmudist(iPID,TrkParam,muarr);

    qobs.assign(wc->NPMT(), 0.);

    double trigOffset = trigWCSim->GetHeader()->GetDate();

    //    std::cout << "PCflg" << PCflg << " fPCflcgut" << fPCflcgut << std::endl;
    if (PCflg == 0 || fPCflcgut == 0) {//FC event
     
      //      std::cout << "Ndigihits" << trigWCSim->GetNcherenkovdigihits() <<std::endl;

      for (int ih = 0; ih < trigWCSim->GetNcherenkovdigihits(); ih++){
	WCSimRootCherenkovDigiHit * digiHit = dynamic_cast<WCSimRootCherenkovDigiHit*>( trigWCSim->GetCherenkovDigiHits()->At(ih) );

	int icab = digiHit->GetTubeId()-1;
	//	cout << "trigWCSim->GetCherenkovHits()->GetEntries() " << trigWCSim->GetCherenkovHits()->GetEntries()  << " trigWCSim->GetCherenkovHitTimes()->GetEntries() " << trigWCSim->GetCherenkovHitTimes()->GetEntries() << endl;

	WCSimRootPMT pmtObj = geomWCSim->GetPMT(icab);

	//	cout << " digiHit->GetTubeId() " << digiHit->GetTubeId() << " pmtObj.GetTubeNo() " << pmtObj.GetTubeNo() << endl;
	
	//	if (icab <= 1 || icab >=11145) cout << icab << endl;
	
	TVector3 wcPmtPos(pmtObj.GetPosition(0), pmtObj.GetPosition(1), pmtObj.GetPosition(2));

	double RmidPMT = sqrt ( pow (  fiTQun_shared::Get()->TfmDetToGlblCoord(wcPmtPos, true, true).X() - midpos[0], 2) +
				pow (  fiTQun_shared::Get()->TfmDetToGlblCoord(wcPmtPos, true, true).Y() - midpos[1], 2) +
				pow (  fiTQun_shared::Get()->TfmDetToGlblCoord(wcPmtPos, true, true).Z() - midpos[2], 2) );
	
	if (!aSubToffs) aSubToffs = 950 - trigOffset;

	double tc = digiHit->GetT() - aSubToffs - TrkParam[3] - RmidPMT*nwtr/c0 - smid/c0;// TrkParam[3] = truth interaction time: the reference subtracts it (makehist.cc:199) for samples not generated at t=0; LUCiD randomises it per event.

	// Fill histogram
	//	std::cout << "tc" << tc <<" logmu" << log10(muarr[icab]) << std::endl;

	htimepdf.Fill(tc,log10(muarr[icab]));

	// charge-vs-distance calibration: everything needed to set WAttenL by
	// matching prediction to observation, which is the only thing it controls.
	qR_R    = RmidPMT;
	qR_mu   = muarr[icab];
	qR_q    = digiHit->GetQ();
	qR_mom  = TrkParam[6];
	qR_cos  = 0.;
	qrtree->Fill();
	hsum_q.Fill(RmidPMT, digiHit->GetQ());
	qobs[icab] += digiHit->GetQ();
      }

      // Now that qobs holds this event's observed charge, compare it with the
      // prediction over EVERY PMT. hsum_* is the unrestricted ratio; hsig_* is
      // restricted to illuminated PMTs, because the dark floor dominates the
      // unrestricted ratio at large R and the model rightly does not predict it.
      for (int ip = 0; ip < wc->NPMT(); ip++) {
        WCSimRootPMT pm = geomWCSim->GetPMT(ip);
        TVector3 pv(pm.GetPosition(0), pm.GetPosition(1), pm.GetPosition(2));
        TVector3 pg = fiTQun_shared::Get()->TfmDetToGlblCoord(pv, true, true);
        double dR = sqrt(pow(pg.X()-midpos[0],2)+pow(pg.Y()-midpos[1],2)
                        +pow(pg.Z()-midpos[2],2));
        hsum_mu.Fill(dR, muarr[ip]);
        if (muarr[ip] > 0.5) {
          hsig_mu.Fill(dR, muarr[ip]);
          hsig_q .Fill(dR, qobs[ip]);
        }
      }
    }
    else {
      std::cout << "PC event!!" << std::endl;
    }
  }

  

  TFile *ofile = new TFile(ntplFile + "_hist.root", "RECREATE");
  htimepdf.Write();
  qrtree->Write();
  hsum_q.Write();
  hsum_mu.Write();
  hsig_q.Write();
  hsig_mu.Write();
  delete ofile;


}
  
