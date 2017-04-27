#define _GNU_SOURCE
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <sched.h>


unsigned long long timespecDiff(struct timespec *timeA_p, struct timespec *timeB_p)
{
  return ((timeA_p->tv_sec * 1000000000) + timeA_p->tv_nsec) -
           ((timeB_p->tv_sec * 1000000000) + timeB_p->tv_nsec);
}


int main(void)
{
        int     fd_1[2], fd_2[2];
        char    string_child[100]; 
        char    string_parent[100]; 
        char    readbuffer[5]; 
        cpu_set_t set;
	pid_t   child_pid;
        int pCPU = 0, cCPU = 0; 

        struct timespec start,stop;
        unsigned long long result;

        CPU_ZERO(&set); 

    
        if(pipe(fd_1) == -1 || pipe(fd_2) == -1) 
        {
            perror("pipe closed");
            exit(1);
        }

        if((child_pid = fork()) == -1) 
        {
            perror("fork closed");
            exit(1);
        }

        
        if(child_pid == 0) 
        {
            CPU_SET(cCPU, &set); 
            int i;

            if (sched_setaffinity(0, sizeof(set), &set) == -1) 
                perror("sched set affinity not set");
 
            close(fd_1[1]); 
            close(fd_2[0]); 


            while(read(fd_1[0],readbuffer,sizeof(readbuffer)) > 0) 
            {
                write(fd_2[1], string_child, strlen(string_child)+1);
            } 

            exit(0);
        }   
else 
        {
            int i =0 ;
            unsigned long long sum = 0;

            CPU_SET(pCPU, &set);

            if (sched_setaffinity(0, sizeof(set), &set) == -1) 
                perror("sched set affinity not set");
            
            while ( i < 200) {

                close(fd_1[0]);
                close(fd_2[1]); 

                if(write(fd_1[1], string_parent, strlen(string_parent)+1) != -1) 
                {
                    clock_gettime(CLOCK_MONOTONIC, &start);
                    read(fd_2[0], readbuffer, sizeof(readbuffer));
                    clock_gettime(CLOCK_MONOTONIC, &stop);
                    sum = sum + timespecDiff(&stop,&start);
          
                }
                i++;

            }
            printf("\nProcess switch time : %llu\n",sum/(2*200));
        }




        return(0);
}

