#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <stdbool.h>
#include <semaphore.h>
#include <time.h>
#include <assert.h>

pthread_mutex_t lock;
int num=0;

void* numOne(void *p){
	
	pthread_mutex_lock(&lock);
	while (num) {
		num=0;
		//printf("thread1: %d ", num);
	}
	pthread_mutex_unlock(&lock);
	
	return NULL;
}

void* numZero(void *p){
	
	pthread_mutex_lock(&lock);
	
	while(!num) {
		num=1;
		//printf("thread2: %d ", num);
	}
	pthread_mutex_unlock(&lock);
	
	return NULL;
}


int main(){
	
	struct timespec start, end; //Initialize time
	
	pthread_t thread1;
	pthread_t thread2;
	
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start); //start timer
	
	//pthread_create(&thread1, NULL, numOne , NULL);
	//pthread_create(&thread2, NULL, numZero , NULL);
	
	//sleep(1);
	
	//pthread_join(thread1, NULL);
	//pthread_join(thread2, NULL);
	
	//end=clock();

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end); //end timer
	
	double result = (end.tv_sec - start.tv_sec)* 1000 + (end.tv_nsec - start.tv_nsec) * 0.0001; //calculate time
	printf("Minimal function call for thread :%f \n", result );
	
	return 0;
}
