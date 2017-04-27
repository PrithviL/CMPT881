#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//for calculation of time elapsed
unsigned long long timespecDiff(struct timespec *timeA_p, struct timespec *timeB_p)
{
  return ((timeA_p->tv_sec * 1000000000) + timeA_p->tv_nsec) -
           ((timeB_p->tv_sec * 1000000000) + timeB_p->tv_nsec);
}

int main(void)
{
    struct timespec start;
    struct timespec stop;
    unsigned long long average; 
    int i = 0, sum = 0;

    start.tv_sec = 0, stop.tv_sec = 0;
    start.tv_nsec = 0, stop.tv_nsec = 0;

    while(i < 1000000) {
       
        clock_gettime(CLOCK_MONOTONIC, &start);
        getpid();
        clock_gettime(CLOCK_MONOTONIC, &stop);

     
        sum += timespecDiff(&stop,&start);
        i++;

    }
    average = sum/1000000;
    printf("Minimal sys call cost : %llu\n",average);

    return 0;

}
