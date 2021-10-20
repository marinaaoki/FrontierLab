# Fall risk detection system
This project was developed for the FrontierLab@Osaka program at Osaka University in the 2021 Spring/Summer term under the supervision of Prof. Maekawa at the Hara Laboratory. 

The system detects potentially dangerous objects in frequently-used paths in videos recorded within homes and aims to reduce the risk of falls for older adults while considering information disclosure. The aim of the research was to discover the greatest difficulties faced by older adults and their informal caregivers during their communication with one another in their daily lives. To do so, I analysed interviews with informal caregivers using the SCAT method and used the results to develop design guidelines that would facilititate information acquistion for the remote caregivers of independently lliving eldelry.

# Technologies used
* Python
  * [OpenCV](https://github.com/opencv/opencv-python "OpenCV")
  * [Simpful](https://github.com/aresio/simpful "Simpful")
* [M5Camera](https://github.com/m5stack/M5Stack-Camera "M5Camera")

The assumed environment for the system involves a camera that is installed to monitor the floor and the movements of a person within the scene. The proposed vision-based detection system uses the M5Camera by M5Stack. The M5Camera is a development board using the ESP32 chip and an OV2640 camera module. It comes with a preloaded software that is programmed using the ESP-IDF framework developed by Espressif Systems. The software allows the M5Camera to function as an access point which the user can connect to in order to view the video stream. The video stream is then processed on a computer after saving the current frame from the specified URL.

# Object Detection Algorithm

