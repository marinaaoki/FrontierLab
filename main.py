from __future__      import print_function
from multiprocessing import Process
from time            import sleep
from stream          import create_stream
from detect          import detect_people
from helper          import save_video

import os

if __name__ == '__main__':
    source = 'C:\\Users\\aokim\\Documents\\Bachelorarbeit\\opencv\\data'
    folder = "C:\\Users\\aokim\\Documents\\Bachelorarbeit\\opencv\\results\\result_final_0.30"

    #p1 = Process(target=create_stream)
    #p1.start()

    # TODO: Instead of giving static head start, enclose in while loop so while there is a next frame to process, continue or wait.
    # Give the stream a head start of 30 seconds.
    #sleep(30)

    #p2 = Process(target=detect_people)
    #p2.start()

    #create_stream(source)
    #save_video(source)
    detect_people(folder, source)
    #save_video(folder)
