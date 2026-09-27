#ifndef TPolyFunc_h_seen
#define TPolyFunc_h_seen

#include "TF1.h"
#include "TString.h"

class TPolyFunc : public TF1 {

 public:
  TPolyFunc();
  TPolyFunc(TString name, double xmin, double xmax, int order0, int order1=-1, int order2=-1, int order3=-1, int order4=-1, int order5=-1, int order6=-1, int order7=-1, int order8=-1, int order9=-1);
  ~TPolyFunc();

  void Initialize(TString name, double xmin, double xmax, int order0, int order1=-1, int order2=-1, int order3=-1, int order4=-1, int order5=-1, int order6=-1, int order7=-1, int order8=-1, int order9=-1);

  void PrintSubPolys(bool printDerivatives=false);
  void SetSubParameters(int lastPolyToSet, const Double_t* par=0);
  //  virtual void Copy(TObject& f1);

 protected:

  static int CalcNPar(int order0, int order1=-1, int order2=-1, int order3=-1, int order4=-1, int order5=-1, int order6=-1, int order7=-1, int order8=-1, int order9=-1);

  Double_t myPoly(const Double_t *x, const Double_t *par);
  static Double_t Zero(Double_t *x, Double_t *par);

  int nsubpolys;
  TF1* subpolys[10];
  TF1* subpolyders[10];

  //  std::vector<TF1> subpolys;
  //  static int nTPolyFuncs;

  ClassDef(TPolyFunc,1);

};

#endif
