/* Full output equality against kernelpool's fused bitonic router. */
#include "../ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL line%d draw%u arm%u\n",__LINE__,draw,arm);return 1;}}while(0)
static uint32_t seed=297731;
static uint32_t rnd(void){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return seed;}
static float rf(void){return (int32_t)rnd()*0x1p-30f;}
int main(int argc,char **argv){
 unsigned draw=0,arm=0,draws=argc>1?strtoul(argv[1],NULL,10):32;
 void *map=NULL;const size_t bytes=4096;CHECK(!posix_memalign(&map,getpagesize(),bytes));memset(map,0,bytes);
 float *bias=map;CHECK(ds4_gpu_init()&&ds4_gpu_set_model_map(map,bytes));
 ds4_gpu_tensor *in=ds4_gpu_tensor_alloc(384*4),*out[3];const unsigned n[]={384,6,6};
 uint32_t ref[3][400],got[3][400],poison[400];for(unsigned i=0;i<400;i++)poison[i]=0x7fc12345;
 for(unsigned i=0;i<3;i++){out[i]=ds4_gpu_tensor_alloc((n[i]+16)*4);CHECK(out[i]);}
 const int edges=argc>2;
 float logits[384];uint64_t words=0;unsigned changed=0;
 for(draw=0;draw<draws+1000;draw++){
  int repeat=draw>=draws;
  if(!repeat){for(unsigned i=0;i<384;i++){logits[i]=rf()*16;bias[i]=rf()*.1f;}
   if(draw%11==0)for(unsigned i=0;i<384;i++){logits[i]=1;bias[i]=0;} /* tie fallback */
   if(draw%13==0)for(unsigned i=0;i<384;i++){logits[i]=(float)(i%6);bias[i]=0;}
   if(edges){unsigned j=draw%384;switch(draw%6){case 0:logits[j]=NAN;break;case 1:logits[j]=INFINITY;break;case 2:logits[j]=-INFINITY;break;case 3:bias[j]=NAN;break;case 4:bias[j]=INFINITY;break;case 5:bias[j]=-INFINITY;break;}}
  }
  CHECK(ds4_gpu_tensor_write(in,0,logits,sizeof(logits)));
  for(arm=repeat?1:0;arm<(repeat?2:3);arm++){
   if(arm==2){for(unsigned i=0;i<384;i++)logits[i]=logits[i]*1.25f+2;CHECK(ds4_gpu_tensor_write(in,0,logits,sizeof(logits)));}
   if(arm==0)setenv("DS4_METAL_DISABLE_V41_ROUTER_HIER","1",1);else unsetenv("DS4_METAL_DISABLE_V41_ROUTER_HIER");
   for(unsigned i=0;i<3;i++)CHECK(ds4_gpu_tensor_write(out[i],0,poison,(n[i]+16)*4));
   CHECK(ds4_gpu_begin_commands()&&ds4_gpu_dsv41_router_one(out[1],out[2],out[0],in,map,bytes,0,384,6,1.5f)&&ds4_gpu_end_commands());
   int delta=0;for(unsigned i=0;i<3;i++){
    uint32_t *dst=arm?got[i]:ref[i];CHECK(ds4_gpu_tensor_read(out[i],0,dst,(n[i]+16)*4)&&!memcmp(dst+n[i],poison+n[i],64));
    for(unsigned j=0;j<n[i];j++){if(i==1)CHECK(dst[j]<384u);else{float x;memcpy(&x,dst+j,4);CHECK(edges||isfinite(x));}}
    if(arm==1){if(memcmp(ref[i],got[i],n[i]*4)){for(unsigned j=0;j<n[i];j++)if(ref[i][j]!=got[i][j]){fprintf(stderr,"Mismatch tensor%u word%u %08x %08x\n",i,j,ref[i][j],got[i][j]);break;}return 1;}if(!repeat)words+=n[i];}
    if(arm==2){delta|=memcmp(ref[i],got[i],n[i]*4)!=0;memcpy(ref[i],got[i],n[i]*4);}
   }if(arm==2){CHECK(delta);changed++;}
  }
  if(!repeat&&(draw+1)%1000==0){printf("%u/%u exact\n",draw+1,draws);fflush(stdout);}
 }
 printf("PASS draws=%u words=%llu controls=%u repeats=1000\n",draws,(unsigned long long)words,changed);return 0;
}
