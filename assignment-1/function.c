    #include <stdio.h>
#include <stdlib.h>
#include <time.h>

unsigned long long timespecDiff(struct timespec *timeA_p, struct timespec *timeB_p)
{
  return ((timeA_p->tv_sec * 1000000000) + timeA_p->tv_nsec) -
           ((timeB_p->tv_sec * 1000000000) + timeB_p->tv_nsec);
}

//creating an empty function that does nothing
void do_nothing(void)
{
    //do nothing
}

int main(void)
{
    struct timespec start;
    struct timespec stop;
    double average; //64 bit integer
    int i = 0, sum = 0;

    start.tv_sec = 0, stop.tv_sec = 0;  //starts and stops timer in seconds
    start.tv_nsec = 0, stop.tv_nsec = 0; //starts and stops timer in nanoseconds

  
    while(i < 1000000) {           // 1000000 times it loops to find the average time of function call
        clock_gettime(CLOCK_MONOTONIC, &start);  
        do_nothing();
        clock_gettime(CLOCK_MONOTONIC, &stop);
        sum += timespecDiff(&stop,&start);      //it increments evertime to calculate the difference
        i++;

    }

    //report average
    average = sum/1000000;
    printf("Minimal function call cost : %f \n",average);

    return 0;

}
