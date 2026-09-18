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
  CHECK(ds4_gpu_init());unsigned k=shapes[shape][0],n=shapes[shape][1];
  size_t wb=(size_t)k/32*n*sizeof(block),sz=wb+256,ob=(n+16)*sizeof(float);
  void *map=NULL;CHECK(!posix_memalign(&map,getpagesize(),sz));memset(map,0,sz);
  block *w=(block*)((char*)map+256);
  for(size_t i=0;i<wb/sizeof(block);i++){w[i].d=(uint16_t)(0x1000+rnd()%0x6000);for(unsigned q=0;q<32;q++)w[i].qs[q]=(int8_t)rnd();}
  CHECK(ds4_gpu_set_model_map(map,sz));
  ds4_gpu_tensor *in=ds4_gpu_tensor_alloc(k*4),*out=ds4_gpu_tensor_alloc(ob);CHECK(in&&out);
  float *x=malloc(k*4),*ref=malloc(ob),*got=malloc(ob),*poison=malloc(ob);CHECK(x&&ref&&got&&poison);
  for(unsigned i=0;i<n+16;i++){uint32_t v=i<n?0x7fc12345:0xdeadbeef;memcpy(poison+i,&v,4);}
  for(unsigned io=0;io<2;io++){
   for(unsigned d=0;d<draws;d++){
    for(unsigned i=0;i<k;i++)x[i]=(int32_t)rnd()*0x1p-27f;
    CHECK(ds4_gpu_tensor_write(in,0,x,k*4));
    for(unsigned arm=0;arm<3;arm++){
     if(arm==0)setenv("DS4_METAL_DISABLE_V41_Q8_SHORT","1",1);else unsetenv("DS4_METAL_DISABLE_V41_Q8_SHORT");
     if(arm==2){for(unsigned i=0;i<k;i++)x[i]=x[i]*1.5f+.125f;CHECK(ds4_gpu_tensor_write(in,0,x,k*4));}
     CHECK(ds4_gpu_tensor_write(out,0,poison,ob));CHECK(ds4_gpu_begin_commands());
     CHECK(io?ds4_gpu_matmul_q8_0_bf16io_tensor(out,map,sz,256,k,n,in,1):ds4_gpu_matmul_q8_0_bf16_tensor(out,map,sz,256,k,n,in,1));
     CHECK(ds4_gpu_end_commands());CHECK(ds4_gpu_tensor_read(out,0,arm?got:ref,ob));
     float *p=arm?got:ref;CHECK(!memcmp(p+n,poison+n,16*4));for(unsigned i=0;i<n;i++)CHECK(isfinite(p[i]));
     if(arm==1){if(memcmp(ref,got,n*4)){for(unsigned i=0;i<n;i++)if(memcmp(ref+i,got+i,4)){fprintf(stderr,"Mismatch shape%u io%u draw%u row%u %.9g %.9g\n",shape,io,d,i,ref[i],got[i]);break;}return 1;}words+=n;}
     if(arm==2){CHECK(memcmp(ref,got,n*4));controls++;}
    }
    if((d+1)%1000==0){printf("shape%u io%u %u/%u exact\n",shape,io,d+1,draws);fflush(stdout);}
   }
   /* Last output is the changed-input control; repeat those identical inputs. */
   memcpy(ref,got,ob);
   for(unsigned repeat=0;repeat<1000;repeat++){
    CHECK(ds4_gpu_tensor_write(out,0,poison,ob));CHECK(ds4_gpu_begin_commands());
    CHECK(io?ds4_gpu_matmul_q8_0_bf16io_tensor(out,map,sz,256,k,n,in,1):ds4_gpu_matmul_q8_0_bf16_tensor(out,map,sz,256,k,n,in,1));
    CHECK(ds4_gpu_end_commands()&&ds4_gpu_tensor_read(out,0,got,ob));CHECK(!memcmp(ref,got,ob));
   }
   printf("PASS shape%u io%u draws%u repeats1000\n",shape,io,draws);fflush(stdout);
  }
  ds4_gpu_tensor_free(in);ds4_gpu_tensor_free(out);ds4_gpu_cleanup();free(map);free(x);free(ref);free(got);free(poison);
 }
 printf("PASS exact_words=%llu controls=%llu\n",(unsigned long long)words,(unsigned long long)controls);return 0;
}
