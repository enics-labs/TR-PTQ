# create database for post simulation waveform
database -open -shm dump -compress -incsize 100M -into dump -default
probe -database dump -create [scop -tops] -functions -tasks -emptyok -depth all -memories -all

#run until $finish signal and exit
run 100000
exit
