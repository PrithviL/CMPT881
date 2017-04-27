#include <stdio.h>
#include <stdint.h>
#include <time.h>

// timespecDiff Function gives the elapsed time from the timer start time till it is stopped.
// timespec has two varialbes one has time in seconds and other in nanoseconds.
// Result of the difference is given in nanoseconds

unsigned long long timespecDiff(struct timespec *timeA_p, struct timespec *timeB_p)
{
  return ((timeA_p->tv_sec * 1000000000) + timeA_p->tv_nsec) -
           ((timeB_p->tv_sec * 1000000000) + timeB_p->tv_nsec);
}


int main()
{
struct timespec start; //start of timespec type
struct timespec stop; //stop of timespec type
unsigned long long result; //64 bit integer ; elapsed time is stored here

clock_gettime(CLOCK_REALTIME, &start); //represents the machine's best-guess as to the current wall-clock, time-of-day time and stores in start.
sleep(1); //sleep the current process for 1 sec
clock_gettime(CLOCK_REALTIME, &stop); //represents the machine's best-guess as to the current wall-clock, time-of-day time and stores in stop.
    
result=timespecDiff(&stop,&start); //finds the elapsed time between start and stop

printf("CLOCK_REALTIME Measured: %llu\n",result); //prints the difference when REALTIME clock is used

clock_gettime(CLOCK_MONOTONIC, &start); //represents the absolute elapsed wall-clock time since some arbitrary, fixed point in the past and stores in start
sleep(1);
clock_gettime(CLOCK_MONOTONIC, &stop); //represents the absolute elapsed wall-clock time since some arbitrary, fixed point in the past and stores in stop

result=timespecDiff(&stop,&start);

printf("CLOCK_MONOTONIC Measured: %llu\n",result); //print the difference when MONOTONIC clock is used

clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start); //this clock type will get how much cpu time the process takes and stores in start
sleep(1);
clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);  //this clock type will get how much cpu time the process takes and stores in stop

result=timespecDiff(&stop,&start);

printf("CLOCK_PROCESS_CPUTIME_ID Measured: %llu\n",result); //print the difference when PROCESS_CPUTIME_ID clock is used

clock_gettime(CLOCK_THREAD_CPUTIME_ID, &start); //this clock type will get how much cpu time a thread takes and stores in start
sleep(1);
clock_gettime(CLOCK_THREAD_CPUTIME_ID, &stop); //this clock type will get how much cpu time a thread takes and stores in stop

result=timespecDiff(&stop,&start);

printf("CLOCK_THREAD_CPUTIME_ID Measured: %llu\n",result); //prints the difference when THREAD_CPUTIME_ID clock is used


}



/*CLOCK_REALTIME Measured: 1004515000
 CLOCK_MONOTONIC Measured: 1001890000
 CLOCK_PROCESS_CPUTIME_ID Measured: 48000
 CLOCK_THREAD_CPUTIME_ID Measured: 34388*/

