#include "autograd.hpp"
#include "ops2d.hpp"
#include <cstdio>
#include <fstream>
static std::vector<float> rd(const std::string&p){std::ifstream f(p,std::ios::binary|std::ios::ate);auto n=f.tellg();f.seekg(0);std::vector<float>v(n/4);f.read((char*)v.data(),n);return v;}
static double md(const std::vector<float>&a,const std::vector<float>&b){double m=0;for(size_t i=0;i<a.size();++i)m=std::max(m,(double)std::abs(a[i]-b[i]));return m;}
int main(){
  const std::string D="pure/ref/data_gconv/"; std::ifstream mf(D+"meta.txt"); std::string tag; int C,Cout,g; bool ok=true;
  while(mf>>tag>>C>>Cout>>g){
    auto x=from_data({1,C,10,10},rd(D+tag+"_x.bin"),true);
    auto w=from_data({Cout,C/g,3,3},rd(D+tag+"_w.bin"),true);
    auto b=from_data({Cout},rd(D+tag+"_b.bin"),true);
    auto gy=from_data({1,Cout,10,10},rd(D+tag+"_gy.bin"));
    auto y=conv2d(x,w,b,1,1,g); auto loss=sum(mul(y,gy)); backward(loss);
    double dy=md(y->data,rd(D+tag+"_y.bin")),dx=md(x->grad,rd(D+tag+"_dx.bin")),dw=md(w->grad,rd(D+tag+"_dw.bin")),db=md(b->grad,rd(D+tag+"_db.bin"));
    bool p=dy<1e-4&&dx<1e-4&&dw<1e-4&&db<1e-4; ok=ok&&p;
    printf("[%-3s g=%d] y=%.2e dx=%.2e dw=%.2e db=%.2e %s\n",tag.c_str(),g,dy,dx,dw,db,p?"OK":"FAIL");
  }
  printf("\n%s\n",ok?"grouped conv == torch (fwd+bwd)":"MISMATCH"); return ok?0:1;
}
