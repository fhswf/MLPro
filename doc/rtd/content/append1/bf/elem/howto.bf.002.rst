.. _Howto BF 2:
`Howto 02 - (Various) Timer <https://github.com/fhswf/MLPro/blob/main/examples/bf/Howto%2002%20-%20(Various)%20Timer.py>`_
================
Ver. 1.0.1 (2021-11-13)

This is an example of implementation of Timer class functionality of MLPro. The timer class allows you to operate timer in virtual or real mode and setup lap durations as per required. A snippet of output shows an example of Timer class immplementation. In the examples below the timer runs for 10 laps with 3 random time steps registered in every lap. The timer runs in virtual mode and real mode in example one and example two respectively. In the second example the timer logs a warning whenever the total random steps in a lap extends the total lap duration set. To script for this result is given in the example code section of this document.

Results
`````````````````
This is an example output to understand the Timer class functionality. 

::

	Example 1: Timer in virtual time mode with lap duration 1 day and 15 seconds...
	
	
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:00 , Cycle 0 Lap time 0:00:00 -- Instantiated 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:00 , Cycle 0 Lap time 0:00:00 -- Process step 1 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:00 , Cycle 0 Lap time 0:00:00 -- Process step 1 ended after 0.384717969280453 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:00 , Cycle 0 Lap time 0:00:00 -- Process step 2 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:00 , Cycle 0 Lap time 0:00:00 -- Process step 2 ended after 0.37521495358780044 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:00 , Cycle 0 Lap time 0:00:00 -- Process step 3 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:00 , Cycle 0 Lap time 0:00:00 -- Process step 3 ended after 0.4916414331178196 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 1 day, 0:00:15 , Cycle 1 Lap time 0:00:00 -- Process step 1 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 1 day, 0:00:15 , Cycle 1 Lap time 0:00:00 -- Process step 1 ended after 0.004843486728168788 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 1 day, 0:00:15 , Cycle 1 Lap time 0:00:00 -- Process step 2 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 1 day, 0:00:15 , Cycle 1 Lap time 0:00:00 -- Process step 2 ended after 0.08821206252455716 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 1 day, 0:00:15 , Cycle 1 Lap time 0:00:00 -- Process step 3 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 1 day, 0:00:15 , Cycle 1 Lap time 0:00:00 -- Process step 3 ended after 0.08351858114846646 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 2 days, 0:00:30 , Cycle 2 Lap time 0:00:00 -- Process step 1 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 2 days, 0:00:30 , Cycle 2 Lap time 0:00:00 -- Process step 1 ended after 0.05321112599944315 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 2 days, 0:00:30 , Cycle 2 Lap time 0:00:00 -- Process step 2 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 2 days, 0:00:30 , Cycle 2 Lap time 0:00:00 -- Process step 2 ended after 0.4344094258447032 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 2 days, 0:00:30 , Cycle 2 Lap time 0:00:00 -- Process step 3 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 2 days, 0:00:30 , Cycle 2 Lap time 0:00:00 -- Process step 3 ended after 0.18610618410616336 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 3 days, 0:00:45 , Cycle 3 Lap time 0:00:00 -- Process step 1 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 3 days, 0:00:45 , Cycle 3 Lap time 0:00:00 -- Process step 1 ended after 0.40118406766326214 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 3 days, 0:00:45 , Cycle 3 Lap time 0:00:00 -- Process step 2 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 3 days, 0:00:45 , Cycle 3 Lap time 0:00:00 -- Process step 2 ended after 0.5157481641972429 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 3 days, 0:00:45 , Cycle 3 Lap time 0:00:00 -- Process step 3 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 3 days, 0:00:45 , Cycle 3 Lap time 0:00:00 -- Process step 3 ended after 0.16294340283592074 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 4 days, 0:01:00 , Cycle 4 Lap time 0:00:00 -- Process step 1 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 4 days, 0:01:00 , Cycle 4 Lap time 0:00:00 -- Process step 1 ended after 0.19889498695801602 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 4 days, 0:01:00 , Cycle 4 Lap time 0:00:00 -- Process step 2 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 4 days, 0:01:00 , Cycle 4 Lap time 0:00:00 -- Process step 2 ended after 0.41093390251529655 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 4 days, 0:01:00 , Cycle 4 Lap time 0:00:00 -- Process step 3 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 4 days, 0:01:00 , Cycle 4 Lap time 0:00:00 -- Process step 3 ended after 0.15701579568144772 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 5 days, 0:01:15 , Cycle 5 Lap time 0:00:00 -- Process step 1 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 5 days, 0:01:15 , Cycle 5 Lap time 0:00:00 -- Process step 1 ended after 0.28328548487650507 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 5 days, 0:01:15 , Cycle 5 Lap time 0:00:00 -- Process step 2 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 5 days, 0:01:15 , Cycle 5 Lap time 0:00:00 -- Process step 2 ended after 0.06014700470575367 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 5 days, 0:01:15 , Cycle 5 Lap time 0:00:00 -- Process step 3 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 5 days, 0:01:15 , Cycle 5 Lap time 0:00:00 -- Process step 3 ended after 0.17039014653056217 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 6 days, 0:01:30 , Cycle 6 Lap time 0:00:00 -- Process step 1 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 6 days, 0:01:30 , Cycle 6 Lap time 0:00:00 -- Process step 1 ended after 0.5521216455190366 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 6 days, 0:01:30 , Cycle 6 Lap time 0:00:00 -- Process step 2 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 6 days, 0:01:30 , Cycle 6 Lap time 0:00:00 -- Process step 2 ended after 0.18832420614363216 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 6 days, 0:01:30 , Cycle 6 Lap time 0:00:00 -- Process step 3 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 6 days, 0:01:30 , Cycle 6 Lap time 0:00:00 -- Process step 3 ended after 0.426686386639143 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 7 days, 0:01:45 , Cycle 7 Lap time 0:00:00 -- Process step 1 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 7 days, 0:01:45 , Cycle 7 Lap time 0:00:00 -- Process step 1 ended after 0.5239292239724854 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 7 days, 0:01:45 , Cycle 7 Lap time 0:00:00 -- Process step 2 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 7 days, 0:01:45 , Cycle 7 Lap time 0:00:00 -- Process step 2 ended after 0.14217458515818407 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 7 days, 0:01:45 , Cycle 7 Lap time 0:00:00 -- Process step 3 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 7 days, 0:01:45 , Cycle 7 Lap time 0:00:00 -- Process step 3 ended after 0.22553846626460639 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 8 days, 0:02:00 , Cycle 8 Lap time 0:00:00 -- Process step 1 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 8 days, 0:02:00 , Cycle 8 Lap time 0:00:00 -- Process step 1 ended after 0.23877613526860916 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 8 days, 0:02:00 , Cycle 8 Lap time 0:00:00 -- Process step 2 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 8 days, 0:02:00 , Cycle 8 Lap time 0:00:00 -- Process step 2 ended after 0.32923153116805637 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 8 days, 0:02:00 , Cycle 8 Lap time 0:00:00 -- Process step 3 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 8 days, 0:02:00 , Cycle 8 Lap time 0:00:00 -- Process step 3 ended after 0.29734915212982277 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 9 days, 0:02:15 , Cycle 9 Lap time 0:00:00 -- Process step 1 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 9 days, 0:02:15 , Cycle 9 Lap time 0:00:00 -- Process step 1 ended after 0.550721840775017 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 9 days, 0:02:15 , Cycle 9 Lap time 0:00:00 -- Process step 2 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 9 days, 0:02:15 , Cycle 9 Lap time 0:00:00 -- Process step 2 ended after 0.16129622811730085 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 9 days, 0:02:15 , Cycle 9 Lap time 0:00:00 -- Process step 3 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 9 days, 0:02:15 , Cycle 9 Lap time 0:00:00 -- Process step 3 ended after 0.31264457117531286 seconds 
	
	
	
	Example 2: Timer in real time mode with lap duration 1 second and forced timeout situations...
	
	
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:00 , Cycle 0 Lap time 0:00:00 -- Instantiated 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:00.000920 , Cycle 0 Lap time 0:00:00.000920 -- Process step 1 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:00.442661 , Cycle 0 Lap time 0:00:00.442661 -- Process step 1 ended after 0.44030652134767495 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:00.442661 , Cycle 0 Lap time 0:00:00.442661 -- Process step 2 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:00.923847 , Cycle 0 Lap time 0:00:00.923847 -- Process step 2 ended after 0.4803320165832071 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:00.923847 , Cycle 0 Lap time 0:00:00.923847 -- Process step 3 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:01.459230 , Cycle 0 Lap time 0:00:01.459230 -- Process step 3 ended after 0.5332684972743434 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  W  Demo class Timer: Process time 0:00:01.459230 , Cycle 1 Lap time 0:00:00 -- Last process cycle timed out!! 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:01.459917 , Cycle 1 Lap time 0:00:00.000687 -- Process step 1 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:01.867199 , Cycle 1 Lap time 0:00:00.407969 -- Process step 1 ended after 0.4065491485952895 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:01.867199 , Cycle 1 Lap time 0:00:00.407969 -- Process step 2 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:02.320391 , Cycle 1 Lap time 0:00:00.861161 -- Process step 2 ended after 0.45114911275596153 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:02.320391 , Cycle 1 Lap time 0:00:00.861161 -- Process step 3 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:02.424065 , Cycle 1 Lap time 0:00:00.964835 -- Process step 3 ended after 0.10179375935148129 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:02.460545 , Cycle 2 Lap time 0:00:00 -- Process step 1 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:02.520792 , Cycle 2 Lap time 0:00:00.060247 -- Process step 1 ended after 0.05837497499346395 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:02.520792 , Cycle 2 Lap time 0:00:00.060247 -- Process step 2 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:02.625506 , Cycle 2 Lap time 0:00:00.164961 -- Process step 2 ended after 0.10204846354052217 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:02.625506 , Cycle 2 Lap time 0:00:00.164961 -- Process step 3 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:02.966970 , Cycle 2 Lap time 0:00:00.506425 -- Process step 3 ended after 0.34092786752810117 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:03.461084 , Cycle 3 Lap time 0:00:00 -- Process step 1 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:04.025099 , Cycle 3 Lap time 0:00:00.564281 -- Process step 1 ended after 0.5634480842114887 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:04.025879 , Cycle 3 Lap time 0:00:00.564795 -- Process step 2 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:04.371014 , Cycle 3 Lap time 0:00:00.909930 -- Process step 2 ended after 0.3447556955213908 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:04.371014 , Cycle 3 Lap time 0:00:00.909930 -- Process step 3 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:04.615273 , Cycle 3 Lap time 0:00:01.154189 -- Process step 3 ended after 0.24299170233488293 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  W  Demo class Timer: Process time 0:00:04.615273 , Cycle 4 Lap time 0:00:00 -- Last process cycle timed out!! 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:04.616268 , Cycle 4 Lap time 0:00:00.000995 -- Process step 1 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:04.886669 , Cycle 4 Lap time 0:00:00.271396 -- Process step 1 ended after 0.2697828089508598 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:04.886669 , Cycle 4 Lap time 0:00:00.271396 -- Process step 2 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:05.127687 , Cycle 4 Lap time 0:00:00.512414 -- Process step 2 ended after 0.23890741202316343 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:05.127687 , Cycle 4 Lap time 0:00:00.512414 -- Process step 3 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:05.658136 , Cycle 4 Lap time 0:00:01.042863 -- Process step 3 ended after 0.5295034600139086 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  W  Demo class Timer: Process time 0:00:05.658136 , Cycle 5 Lap time 0:00:00 -- Last process cycle timed out!! 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:05.659142 , Cycle 5 Lap time 0:00:00.001006 -- Process step 1 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:05.706385 , Cycle 5 Lap time 0:00:00.048249 -- Process step 1 ended after 0.046753845483454445 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:05.706385 , Cycle 5 Lap time 0:00:00.048249 -- Process step 2 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:05.834733 , Cycle 5 Lap time 0:00:00.176597 -- Process step 2 ended after 0.12624376772868412 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:05.834733 , Cycle 5 Lap time 0:00:00.176597 -- Process step 3 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:06.364531 , Cycle 5 Lap time 0:00:00.706395 -- Process step 3 ended after 0.5271226987110645 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:06.659286 , Cycle 6 Lap time 0:00:00 -- Process step 1 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:06.742410 , Cycle 6 Lap time 0:00:00.083124 -- Process step 1 ended after 0.08294090111970517 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:06.742410 , Cycle 6 Lap time 0:00:00.083124 -- Process step 2 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:07.061117 , Cycle 6 Lap time 0:00:00.401831 -- Process step 2 ended after 0.3162441964790709 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:07.061117 , Cycle 6 Lap time 0:00:00.401831 -- Process step 3 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:07.088681 , Cycle 6 Lap time 0:00:00.429395 -- Process step 3 ended after 0.026981702444639488 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:07.660055 , Cycle 7 Lap time 0:00:00 -- Process step 1 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:08.246408 , Cycle 7 Lap time 0:00:00.586353 -- Process step 1 ended after 0.5857802290712275 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:08.246408 , Cycle 7 Lap time 0:00:00.586353 -- Process step 2 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:08.713028 , Cycle 7 Lap time 0:00:01.052973 -- Process step 2 ended after 0.46537322365492 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:08.713028 , Cycle 7 Lap time 0:00:01.052973 -- Process step 3 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:09.020035 , Cycle 7 Lap time 0:00:01.359980 -- Process step 3 ended after 0.306086350378808 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  W  Demo class Timer: Process time 0:00:09.020163 , Cycle 8 Lap time 0:00:00 -- Last process cycle timed out!! 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:09.020163 , Cycle 8 Lap time 0:00:00 -- Process step 1 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:09.247560 , Cycle 8 Lap time 0:00:00.227397 -- Process step 1 ended after 0.22640616900564728 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:09.247560 , Cycle 8 Lap time 0:00:00.227397 -- Process step 2 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:09.804243 , Cycle 8 Lap time 0:00:00.784080 -- Process step 2 ended after 0.5554737566030241 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:09.805237 , Cycle 8 Lap time 0:00:00.785074 -- Process step 3 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:10.190266 , Cycle 8 Lap time 0:00:01.170103 -- Process step 3 ended after 0.3835593279407716 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  W  Demo class Timer: Process time 0:00:10.190266 , Cycle 9 Lap time 0:00:00 -- Last process cycle timed out!! 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:10.190266 , Cycle 9 Lap time 0:00:00 -- Process step 1 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:10.293866 , Cycle 9 Lap time 0:00:00.103600 -- Process step 1 ended after 0.10214078586856383 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:10.293866 , Cycle 9 Lap time 0:00:00.103600 -- Process step 2 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:10.414546 , Cycle 9 Lap time 0:00:00.224280 -- Process step 2 ended after 0.11936544499001797 seconds 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:10.414546 , Cycle 9 Lap time 0:00:00.224280 -- Process step 3 started 
	YYYY-MM-DD  HH:MM:SS:ssss  I  Demo class Timer: Process time 0:00:11.012724 , Cycle 9 Lap time 0:00:00.822458 -- Process step 3 ended after 0.5979871139179042 seconds 


Prerequisites
`````````````````

Please install the following packages to run this examples properly:
    - :ref:`MLPro <Installation>`
  ..
    - `Matplotlib <https://pypi.org/project/matplotlib/>`_
  ..
    - `NumPy <https://pypi.org/project/numpy/>`_
  ..
    - `Pytorch <https://pypi.org/project/torch/>`_
  ..
    - `OpenAI Gym <https://pypi.org/project/gym/>`_
  ..
    - `PettingZoo <https://pypi.org/project/PettingZoo/>`_
  ..
    - `Stable-Baselines3 <https://pypi.org/project/stable-baselines3/>`_
  ..
    - `Optuna <https://pypi.org/project/optuna/>`_
  ..
    - `Hyperopt <https://pypi.org/project/hyperopt/>`_
  ..
    - `ROS <http://wiki.ros.org/noetic/Installation>`_
    

Example Code
`````````````````
Run this code to get the results as shown above.

.. literalinclude:: ../../../../../../examples/bf/Howto 02 - (Various) Timer.py
    :language: python

