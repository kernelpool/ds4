/* Complete production-shape byte comparisons, poisoned outputs, guards,
 * changed-input controls and deterministic repeats. No model file needed. */
#include "../ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
typedef struct { uint16_t d; int8_t qs[32]; } block;
static uint32_t rng=7919;
static uint32_t rnd(void){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}
#define CHECK(x) do {if(!(x)){fprintf(stderr,"FAIL line %d\n",__LINE__);return 1;}} while(0)
int main(int argc,char **argv){
 unsigned draws=argc>1?(unsigned)strtoul(argv[1],NULL,10):256;
 const unsigned shapes[][2]={{1280,32768},{2304,5120}};
 uint64_t words=0,controls=0;
 for(unsigned shape=0;shape<2;shape++){
  CHECK(ds4_gpu_init());unsigned k=shapes[shape][0],n=shapes[shape][1],on=shape?n*4:n;
  size_t wb=(size_t)k/32*n*sizeof(block),sz=wb+256,ob=(on+16)*sizeof(float);
  void *map=NULL;CHECK(!posix_memalign(&map,getpagesize(),sz));memset(map,0,sz);
  block *w=(block*)((char*)map+256);
  for(size_t i=0;i<wb/sizeof(block);i++){w[i].d=(uint16_t)(0x1000+rnd()%0x6000);for(unsigned q=0;q<32;q++)w[i].qs[q]=(int8_t)rnd();}
  CHECK(ds4_gpu_set_model_map(map,sz));
  ds4_gpu_tensor *in=ds4_gpu_tensor_alloc(k*4),*out=ds4_gpu_tensor_alloc(ob);CHECK(in&&out);
  float *x=malloc(k*4),*ref=malloc(ob),*got=malloc(ob),*poison=malloc(ob);CHECK(x&&ref&&got&&poison);
  for(unsigned i=0;i<on+16;i++){uint32_t v=i<on?0x7fc12345:0xdeadbeef;memcpy(poison+i,&v,4);}
  ds4_gpu_tensor *residual=ds4_gpu_tensor_alloc(n*16),*add=ds4_gpu_tensor_alloc(n*4),*split=ds4_gpu_tensor_alloc(24*4);
  float *r=malloc(n*16),*a=malloc(n*4),h[24];CHECK(residual&&add&&split&&r&&a);
  unsigned pos=0;
  for(unsigned io=2;io<4;io++){
   for(unsigned d=0;d<draws;d++){
    for(unsigned i=0;i<k;i++)x[i]=(int32_t)rnd()*0x1p-27f;
    CHECK(ds4_gpu_tensor_write(in,0,x,k*4));
    pos=(d*7919u)%393216u;
    for(unsigned i=0;i<n*4;i++)r[i]=(int32_t)rnd()*0x1p-29f;
    for(unsigned i=0;i<n;i++)a[i]=(int32_t)rnd()*0x1p-29f;
    for(unsigned i=0;i<24;i++)h[i]=(int32_t)rnd()*0x1p-31f;
    CHECK(ds4_gpu_tensor_write(residual,0,r,n*16)&&ds4_gpu_tensor_write(add,0,a,n*4)&&ds4_gpu_tensor_write(split,0,h,24*4));
    for(unsigned arm=0;arm<3;arm++){
     if(arm==0)setenv("DS4_METAL_DISABLE_V41_Q8_SHORT","1",1);else unsetenv("DS4_METAL_DISABLE_V41_Q8_SHORT");
     if(arm==2){for(unsigned i=0;i<k;i++)x[i]=x[i]*1.5f+.125f;CHECK(ds4_gpu_tensor_write(in,0,x,k*4));}
     CHECK(ds4_gpu_tensor_write(out,0,poison,ob));CHECK(ds4_gpu_begin_commands());
     CHECK(shape?ds4_gpu_dsv41_matmul_expand(out,map,sz,256,k,n,in,io==2?add:NULL,residual,split,4):ds4_gpu_dsv41_project_q(out,map,sz,256,k,n,in,pos,io==3));
     CHECK(ds4_gpu_end_commands());CHECK(ds4_gpu_tensor_read(out,0,arm?got:ref,ob));
     float *p=arm?got:ref;CHECK(!memcmp(p+on,poison+on,16*4));for(unsigned i=0;i<on;i++)CHECK(isfinite(p[i]));
     if(arm==1){if(memcmp(ref,got,on*4)){for(unsigned i=0;i<on;i++)if(memcmp(ref+i,got+i,4)){fprintf(stderr,"Mismatch shape%u io%u draw%u row%u %.9g %.9g\n",shape,io,d,i,ref[i],got[i]);break;}return 1;}words+=on;}
     if(arm==2){CHECK(memcmp(ref,got,on*4));controls++;}
    }
    if((d+1)%1000==0){printf("shape%u io%u %u/%u exact\n",shape,io,d+1,draws);fflush(stdout);}
   }
   /* Last output is the changed-input control; repeat those identical inputs. */
   memcpy(ref,got,ob);
   for(unsigned repeat=0;repeat<1000;repeat++){
    CHECK(ds4_gpu_tensor_write(out,0,poison,ob));CHECK(ds4_gpu_begin_commands());
    CHECK(shape?ds4_gpu_dsv41_matmul_expand(out,map,sz,256,k,n,in,io==2?add:NULL,residual,split,4):ds4_gpu_dsv41_project_q(out,map,sz,256,k,n,in,pos,io==3));
    CHECK(ds4_gpu_end_commands()&&ds4_gpu_tensor_read(out,0,got,ob));CHECK(!memcmp(ref,got,ob));
   }
   printf("PASS shape%u io%u draws%u repeats1000\n",shape,io,draws);fflush(stdout);
  }
  ds4_gpu_tensor_free(residual);ds4_gpu_tensor_free(add);ds4_gpu_tensor_free(split);free(r);free(a);
  ds4_gpu_tensor_free(in);ds4_gpu_tensor_free(out);ds4_gpu_cleanup();free(map);free(x);free(ref);free(got);free(poison);
 }
 printf("PASS exact_words=%llu controls=%llu\n",(unsigned long long)words,(unsigned long long)controls);return 0;
}
