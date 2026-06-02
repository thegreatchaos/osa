#include <stdio.h>
#include <cm/cm.h>
#define DIE_IF(x) do{if(x){fprintf(stderr, "%s failure. %s:%d\n", #x, __FILE__, __LINE__); exit(-__LINE__);}}while(0)

void bench(int m, int n, int k);
int main(int argc, char** argv){
    if(4 != argc){
	fprintf(stderr, "Usage:\n\t%s M K N\n", argv[0]);
	return -__LINE__;
    }
    const int m = (const int)atoi(argv[1]);
    const int k = (const int)atoi(argv[2]);
    const int n = (const int)atoi(argv[3]);

    DIE_IF(0 >= m || 0 >= k || 0 >= n);

    bench(m, n, k);
    return 0;
}

void bench(int m, int n, int k){
    CmDevice* dev;

    CreateCmDevice(dev);
}
