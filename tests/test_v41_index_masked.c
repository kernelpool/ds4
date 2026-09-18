/* Preserve kernelpool wide arithmetic on admitted rows; discarded rows are -inf. */
#include "../ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL line%d draw%u arm%u rows%u\n",__LINE__,draw,arm,rows);return 1;}}while(0)
extern void ds4_gpu_glm_indexer_score_one_force_wide(int);
static uint32_t seed=671337;
static uint32_t rnd(void){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return seed;}
static float rf(void){return (int32_t)rnd()*0x1p-30f;}
int main(int argc,char **argv){
 unsigned draw=0,arm=0,rows=0,draws=argc>1?strtoul(argv[1],NULL,10):32;
 const unsigned cap=300019,blocks=(cap+7)/8;
 float *key=malloc((size_t)cap*128*4),*mask=malloc(blocks*4),*ref=malloc((cap+16)*4),*got=malloc((cap+16)*4),*poison=malloc((cap+16)*4);
 float q[4096],w[32];CHECK(key&&mask&&ref&&got&&poison);
 for(size_t i=0;i<(size_t)cap*128;i++)key[i]=rf();
 for(unsigned i=0;i<cap+16;i++){uint32_t u=0x7fc12345;memcpy(poison+i,&u,4);}
 CHECK(ds4_gpu_init());ds4_gpu_glm_indexer_score_one_force_wide(1);
 ds4_gpu_tensor *kt=ds4_gpu_tensor_alloc((size_t)cap*128*4),*qt=ds4_gpu_tensor_alloc(sizeof(q)),*wt=ds4_gpu_tensor_alloc(sizeof(w)),*mt=ds4_gpu_tensor_alloc(blocks*4),*out=ds4_gpu_tensor_alloc((cap+16)*4);
 CHECK(kt&&qt&&wt&&mt&&out&&ds4_gpu_tensor_write(kt,0,key,(size_t)cap*128*4));
 uint64_t words=0;unsigned changed_controls=0;
 for(draw=0;draw<draws+1000;draw++){
  int repeat=draw>=draws;
  if(!repeat){
   const unsigned geometries[]={16385,32768,65537,131073,262144,300019};
   rows=draw%1000==0?geometries[(draw/1000)%6]:1+rnd()%1024;
   for(unsigned i=0;i<4096;i++)q[i]=rf();for(unsigned i=0;i<32;i++)w[i]=rf();
   for(unsigned i=0;i<(rows+7)/8;i++){unsigned v=rnd()%8;mask[i]=v<4?0.f:v==4?-0.f:v==5?-INFINITY:v==6?NAN:1.f;}
   if(draw%11==0)for(unsigned i=0;i<(rows+7)/8;i++)mask[i]=0.f;
   if(draw%13==0)for(unsigned i=0;i<(rows+7)/8;i++)mask[i]=-INFINITY;
   mask[0]=0.f; /* Keep a surviving row for the changed-input control. */
  }
  CHECK(ds4_gpu_tensor_write(qt,0,q,sizeof(q))&&ds4_gpu_tensor_write(wt,0,w,sizeof(w))&&ds4_gpu_tensor_write(mt,0,mask,((rows+7)/8)*4));
  for(arm=repeat?1:0;arm<(repeat?2:3);arm++){
   if(arm==2){for(unsigned i=0;i<4096;i++)q[i]=q[i]*1.5f+.125f;CHECK(ds4_gpu_tensor_write(qt,0,q,sizeof(q)));}
   CHECK(ds4_gpu_tensor_write(out,0,poison,(rows+16)*4)&&ds4_gpu_begin_commands());
   if(arm==0)CHECK(ds4_gpu_glm_indexer_score_one_tensor(out,qt,wt,kt,rows,32,128,1.f/64,false));
   else CHECK(ds4_gpu_dsv41_indexer_score_masked(out,qt,wt,kt,mt,rows));
   CHECK(ds4_gpu_end_commands());float *dst=arm?got:ref;
   CHECK(ds4_gpu_tensor_read(out,0,dst,(rows+16)*4)&&!memcmp(dst+rows,poison+rows,64));
   for(unsigned i=0;i<rows;i++){
    if(!arm){CHECK(isfinite(dst[i]));if(mask[i/8]!=0.f)dst[i]=-INFINITY;}
    else CHECK(mask[i/8]==0.f?isfinite(dst[i]):dst[i]==-INFINITY);
   }
   if(arm==1){if(memcmp(ref,got,rows*4)){for(unsigned i=0;i<rows;i++)if(memcmp(ref+i,got+i,4)){fprintf(stderr,"Mismatch word%u %.9g %.9g\n",i,ref[i],got[i]);break;}return 1;}if(!repeat)words+=rows;}
   if(arm==2){CHECK(memcmp(ref,got,rows*4));changed_controls++;memcpy(ref,got,rows*4);}
  }
  if(!repeat&&(draw+1)%1000==0){printf("%u/%u exact\n",draw+1,draws);fflush(stdout);}
 }
 printf("PASS draws=%u words=%llu controls=%u repeats=1000\n",draws,(unsigned long long)words,changed_controls);return 0;
}
