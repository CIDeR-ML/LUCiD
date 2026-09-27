#include "TPolyFunc.h"
#include <iostream>

TPolyFunc::TPolyFunc() : TF1(){

}

TPolyFunc::TPolyFunc(TString name, double xmin, double xmax, int order0, int order1, int order2, int order3, int order4, int order5, int order6, int order7, int order8, int order9) : TF1::TF1(name.Data(),this,&TPolyFunc::myPoly,xmin,xmax,CalcNPar(order0,order1,order2,order3,order4,order5,order6,order7,order8,order9),1) {
  //TPolyFunc::TPolyFunc(TString name, double xmin, double xmax, int order0, int order1, int order2, int order3, int order4, int order5, int order6, int order7, int order8, int order9): TF1::TF1() {

  Initialize(name,xmin,xmax,order0,order1,order2,order3,order4,order5,order6,order7,order8,order9);
  //  CalcNPar(order0,order1,order2,order3,order4,order5,order6,order7,order8,order9);

}

void TPolyFunc::Initialize(TString name, double xmin, double xmax, int order0, int order1, int order2, int order3, int order4, int order5, int order6, int order7, int order8, int order9){

  //  CreateFromCintClass(name, this, xmin, xmax, npar, className, 0 );

  TString polynames[10] = {"pol0","pol1","pol2","pol3","pol4","pol5","pol6","pol7","pol8","pol9"};
  int orders[10] = {order0,order1,order2,order3,order4,order5,order6,order7,order8,order9};
  nsubpolys = 0;
  for (int io=0; io<10; io++) {
    if (orders[io] > 9) {
      std::cerr << "ERROR: polynomials are only supported up to 9th order" << std::endl;
      break;
    }
    if (orders[io] >= 0) {
      TString subpolyname = name;
      subpolyname += "_subpoly";
      subpolyname += io;
      TString subpolydername = name;
      subpolydername += "_subpolyder";
      subpolydername += io;
      subpolys[io] = new TF1(subpolyname,polynames[orders[io]],xmin,xmax);
      if (orders[io] > 0) {
	subpolyders[io] = new TF1(subpolydername,polynames[orders[io]-1],xmin,xmax);
      } else {
	subpolyders[io] = new TF1(subpolydername,TPolyFunc::Zero,xmin,xmax,0);
      }
      //      std::cout << "number of parameters for subpoly, subpolyder = " << subpolys[io]->GetNpar() << ", " << subpolyders[io]->GetNpar() << endl;
      nsubpolys++;
    } else {
      break;
    }
  }

}

TPolyFunc::~TPolyFunc(){
}

//TPolyFunc::Copy(TObject& f1){
//  
//}

int TPolyFunc::CalcNPar(int order0, int order1, int order2, int order3, int order4, int order5, int order6, int order7, int order8, int order9){

  int orders[10] = {order0,order1,order2,order3,order4,order5,order6,order7,order8,order9};
  int npar = order0 + 1;
  int nconnections = 0;
  for (int io=1; io<10; io++) {
    if (orders[io] >= 0) {
      npar += orders[io] + 1 - 2; // An nth order polynomial has n+1 parameters, but subtract 2 to make 0th and 1st derivatives agree.
      nconnections++;
    } else {
      break;
    }
    if (orders[io] < 1) {
      std::cerr << "ERROR: lowest polynomial order allowed is 1st (" << orders[io] << ")" << std::endl;
    }
  }

  if (npar < 1) {
    std::cerr << "ERROR: not enough degrees of freedom" << std::endl;
  }

  std::cout << "This TF1 has " << npar + nconnections << " parameters ";
  std::cout << "and the first " << nconnections << " are connection points" << std::endl;
  return npar + nconnections;

}

void TPolyFunc::SetSubParameters(int lastPolyToSet, const Double_t* par){

  if ((lastPolyToSet >= nsubpolys)||(lastPolyToSet < 0)) {
    lastPolyToSet = nsubpolys-1;
  }

  const Double_t* gpars;
  Double_t* owned = nullptr;
  if (par) {
    gpars = par;
  } else {
    owned = new Double_t[GetNpar()];
    for (int ipar=0; ipar<GetNpar(); ipar++) {
      owned[ipar] = GetParameter(ipar);
    }
    gpars = owned;
  }

  int nparused = nsubpolys-1;
  for (int ix=0; ix<=lastPolyToSet; ix++) {
    int ipar_thispoly = nparused -1 + subpolys[ix]->GetNpar();
    if (ix != 0) {
      ipar_thispoly -= 2;
    }
    for (int isubpar=subpolys[ix]->GetNpar()-1; isubpar>=0; isubpar--) {
      // The parameters of the first polynomial are all unconstrained.
      // All subsequent polynomials have parameters 0 and 1 constrained
      // by position and derivative matching.
      if ((ix > 0)&&(isubpar == 0)){
	// Parameter 0 for a constrained polynomial (i.e. not the first polynomial).
	// This parameter is constrained by position matching to the previous polynomial.
	subpolys[ix]->SetParameter(isubpar,0.);
	//	std::cout << "previous poly at " << gpars[ix] << " is " 
	double parval = subpolys[ix-1]->Eval(gpars[ix-1]) - subpolys[ix]->Eval(gpars[ix-1]);
	subpolys[ix]->SetParameter(isubpar,parval);
	//	std::cout << "setting constrained parameter " << isubpar << " in  poly " << ix << " to " << parval << endl;
      } else if ((ix > 0)&&(isubpar == 1)){
	// Parameter 1 for a constrained polynomial (i.e. not the first polynomial).
	// This parameter is constrained by derivative matching to the previous polynomial.
	subpolyders[ix]->SetParameter(isubpar-1,0);
	double parval = subpolyders[ix-1]->Eval(gpars[ix-1]) - subpolyders[ix]->Eval(gpars[ix-1]);
	subpolyders[ix]->SetParameter(isubpar-1,parval);
	subpolys[ix]->SetParameter(isubpar,parval);
	//	std::cout << "setting constrained parameter " << isubpar << " in  poly " << ix << " to " << parval << endl;
      } else {
	// This is an unconstrained parameter. Set it to the next available overall parameter.
	if (isubpar > 0) {
	  subpolyders[ix]->SetParameter(isubpar-1,gpars[ipar_thispoly]*isubpar);
	}
	subpolys[ix]->SetParameter(isubpar,gpars[ipar_thispoly]);
	//	std::cout << "setting unconstrained parameter " << isubpar << " in  poly " << ix << " to " << gpars[ipar_thispoly] << " (global par " << ipar_thispoly << ")" << endl;
	ipar_thispoly--;
	nparused++;
      }
    }
  }
  if (!par) {
    delete [] gpars;
  }
}


Double_t TPolyFunc::myPoly(const Double_t *x, const Double_t *par){

  for (int ix=0; ix<nsubpolys; ix++) {
    if (x[0] < par[ix]) {
      SetSubParameters(ix,par);
      if (subpolys[ix]->Eval(x[0]) > 0) return subpolys[ix]->Eval(x[0]);
      else return 0;
    }
  }

  SetSubParameters(nsubpolys-1,par);
  if (subpolys[nsubpolys-1]->Eval(x[0]) > 0 ) return subpolys[nsubpolys-1]->Eval(x[0]);
  else return 0;

}

Double_t TPolyFunc::Zero(Double_t *x, Double_t *par){
  return 0.;
}

void TPolyFunc::PrintSubPolys(bool printDerivatives){

  TF1** polys;
  if (printDerivatives) {
    polys = subpolyders;
  } else {
    polys = subpolys;
  }

  SetSubParameters(-1);

  for (int ipoly=0; ipoly<nsubpolys; ipoly++) {
    for (int ippar=0; ippar<polys[ipoly]->GetNpar(); ippar++) {
      if (ippar == 0) {
	std::cout << polys[ipoly]->GetParameter(ippar);
      } else if (ippar == 1) {
	std::cout << "  +  " << polys[ipoly]->GetParameter(ippar) << " *x ";
      } else {
	std::cout << "  +  " << polys[ipoly]->GetParameter(ippar) << " *x^" << ippar << " ";
      }
    }
    double min;
    double max;
    if (ipoly == 0) {
      min = GetXmin();
      max = GetParameter(ipoly);
    } else if (ipoly == nsubpolys-1) {
      min = GetParameter(ipoly-1);
      max = GetXmax();
    } else {
      min = GetParameter(ipoly-1);
      max = GetParameter(ipoly);
    }
    std::cout << " (for " << min << " < x < " << max << ")" << endl;
  }
}
