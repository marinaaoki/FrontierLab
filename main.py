from __future__      import print_function
from stream          import create_stream
from detect          import detect_people
from helper          import save_video

if __name__ == '__main__':
    source = "C:\\Users\\aokim\\Documents\\Bachelorarbeit\\opencv\\data\\eval\\video_00"
    folder = "C:\\Users\\aokim\\Documents\\Bachelorarbeit\\opencv\\results\\eval\\17\\0.5"
    disclose_all = False
    threshold = 0.5

    #p1 = Process(target=create_stream)
    #p1.start()

    # Give the stream a head start of 30 seconds.
    #sleep(30)

    #p2 = Process(target=detect_people)
    #p2.start()

    #create_stream(source)
    save_video(source)
    #detect_people(folder, source, disclose_all, threshold)
    #save_video(folder)
