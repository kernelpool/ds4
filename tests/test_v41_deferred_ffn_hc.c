/* Compare the first collapse and original HC with separate norm scratch. */
#include "../ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define CHECK(x) do {if(!(x)){fprintf(stderr,"FAIL line %d draw %u arm %u\n",__LINE__,draw,arm);return 1;}}while(0)
typedef struct {uint16_t d;int8_t q[32];} q8;
static uint32_t seed=91817;
static uint32_t rnd(void){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return seed;}
static float random_float(void){return (int32_t)rnd()*0x1p-30f;}
int main(int argc,char **argv){
 unsigned draw=0,arm=0,draws=argc>1?strtoul(argv[1],NULL,10):32;
 const size_t qo=256,qbytes=384ull*5120*4,ho=qo+qbytes,hbytes=20480ull*24*2;
 const size_t so=ho+hbytes,bo=so+64,no=bo+128,bytes=no+5120*4+256;
 void *map=NULL;CHECK(!posix_memalign(&map,getpagesize(),bytes));memset(map,0,bytes);
 float *qw=(float*)((char*)map+qo);for(unsigned i=0;i<384*5120;i++)qw[i]=random_float()*.02f;
 uint16_t *hw=(uint16_t*)((char*)map+ho);for(unsigned i=0;i<20480*24;i++)hw[i]=(rnd()&0x8000)|(0x2000+rnd()%0x1800);
 float *scale=(float*)((char*)map+so),*base=(float*)((char*)map+bo),*nw=(float*)((char*)map+no);
 scale[0]=.04f;scale[1]=.03f;scale[2]=.02f;
 for(unsigned i=0;i<24;i++)base[i]=random_float()*.1f;
 for(unsigned i=0;i<5120;i++)nw[i]=1.f+random_float()*.1f;
 CHECK(ds4_gpu_init()&&ds4_gpu_set_model_map(map,bytes));
 const unsigned widths[]={5120,5120,24,24,384,5120};ds4_gpu_tensor *out[6];
 float *ref[6],*got[6],*poison[6];
 for(unsigned i=0;i<6;i++){size_t b=(widths[i]+16)*4;out[i]=ds4_gpu_tensor_alloc(b);ref[i]=malloc(b);got[i]=malloc(b);poison[i]=malloc(b);CHECK(out[i]&&ref[i]&&got[i]&&poison[i]);
  for(unsigned j=0;j<widths[i]+16;j++){uint32_t v=j<widths[i]?0x7fc12345:0xdeadbeef;memcpy(poison[i]+j,&v,4);}}
 ds4_gpu_tensor *res=ds4_gpu_tensor_alloc(20480*4),*pre=ds4_gpu_tensor_alloc(16),*qr=ds4_gpu_tensor_alloc(1280*4);CHECK(res&&pre&&qr);
 float *r=malloc(20480*4),x[1280],p[4];CHECK(r);
 uint64_t words=0;
 for(draw=0;draw<draws+1000;draw++){
  const int repeat=draw>=draws;
  if(!repeat){for(unsigned i=0;i<20480;i++)r[i]=random_float();for(unsigned i=0;i<1280;i++)x[i]=random_float();for(unsigned i=0;i<4;i++)p[i]=random_float();}
  CHECK(ds4_gpu_tensor_write(res,0,r,20480*4)&&ds4_gpu_tensor_write(pre,0,p,16)&&ds4_gpu_tensor_write(qr,0,x,1280*4));
  for(arm=repeat?1:0;arm<(repeat?2:3);arm++){
   if(arm==2){for(unsigned i=0;i<20480;i++)r[i]=r[i]*1.5f+.125f;for(unsigned i=0;i<1280;i++)x[i]=x[i]*1.5f+.125f;CHECK(ds4_gpu_tensor_write(res,0,r,20480*4)&&ds4_gpu_tensor_write(qr,0,x,1280*4));}
   for(unsigned i=0;i<6;i++)CHECK(ds4_gpu_tensor_write(out[i],0,poison[i],(widths[i]+16)*4));
   CHECK(ds4_gpu_begin_commands());
   if(arm==0){
    setenv("DS4_METAL_DISABLE_V41_Q8_SHORT","1",1);
    CHECK(ds4_gpu_dsv41_hc_block_input(out[2],out[0],out[1],out[3],res,pre,map,bytes,ho,so,bo,no,20480,24,5120,4,20,1e-6f,1e-6f));
    CHECK(ds4_gpu_matmul_f32_tensor(out[4],map,bytes,qo,5120,384,out[1],1));
    CHECK(ds4_gpu_tensor_copy(out[5],0,out[1],0,5120*4));
   }else{
    CHECK(ds4_gpu_dsv41_deferred_collapse(out[0],out[1],res,pre,map,bytes,no,1e-6f));
    CHECK(ds4_gpu_matmul_f32_tensor(out[4],map,bytes,qo,5120,384,out[1],1));
    CHECK(ds4_gpu_dsv41_hc_block_input(out[2],out[0],out[5],out[3],res,pre,map,bytes,ho,so,bo,no,20480,24,5120,4,20,1e-6f,1e-6f));
   }
   CHECK(ds4_gpu_end_commands());int changed=0;
   for(unsigned i=0;i<6;i++){
    float *dst=arm?got[i]:ref[i];CHECK(ds4_gpu_tensor_read(out[i],0,dst,(widths[i]+16)*4));
    CHECK(!memcmp(dst+widths[i],poison[i]+widths[i],64));for(unsigned j=0;j<widths[i];j++)CHECK(isfinite(dst[j]));
    if(arm==1&&memcmp(ref[i],got[i],widths[i]*4)){
     for(unsigned j=0;j<widths[i];j++)if(memcmp(ref[i]+j,got[i]+j,4)){fprintf(stderr,"Mismatch draw%u tensor%u word%u %.9g %.9g\n",draw,i,j,ref[i][j],got[i][j]);break;}return 1;}
    if(arm==1&&!repeat)words+=widths[i];
    if(arm==2){changed|=memcmp(ref[i],got[i],widths[i]*4)!=0;memcpy(ref[i],got[i],(widths[i]+16)*4);}
   }
   if(arm==2)CHECK(changed);
  }
  if(!repeat&&(draw+1)%1000==0){printf("%u/%u exact\n",draw+1,draws);fflush(stdout);}
 }
 printf("PASS draws=%u words=%llu controls=%u repeats=1000\n",draws,(unsigned long long)words,draws);
 return 0;
}
