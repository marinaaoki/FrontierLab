from __future__      import print_function
from detect          import risk_notification
from helper          import save_video

if __name__ == '__main__':
    source = "/file/to/data/source"
    disclose_all = False
    threshold = 0.3

    risk_notification(source, disclose_all, threshold)
