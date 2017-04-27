all: timer msys mfunction process_switch thread_switch

timer: timer.c
	gcc timer.c -o timer

msys: system.c
	gcc system.c -o msys

mfunction: function.c
	gcc function.c -o mfunction

process_switch: process.c
	gcc process.c -o process_switch

thread_switch: thread.c
	gcc thread_switch.c -lpthread -o thread_switch

clean:
	rm timer msys mfunction process_switch thread_switch 
