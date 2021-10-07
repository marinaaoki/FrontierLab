from __future__      import print_function
from detect          import risk_notification

if __name__ == '__main__':
    source = "/Users/marina/introcs/Bachelorarbeit/opencv/data/eval/video_11"
    folder = "/Users/marina/introcs/Bachelorarbeit/opencv/results/eval/11/0.3"
    disclose_all = False
    threshold = 0.3

    risk_notification(folder, source, disclose_all, threshold)
