# Lane Detection using CNN

This project implements Lane Detection using a Convolutional Neural Network and uses training data from CARLA Simulator. Given an image, the network masks out the lanes alone from the image. The implementation works for all weather conditions even for wet roads in which there may be puddles, etc. which my reflect the sky and cause normal Computer Vision methods to classify them as lanes.

The images shown below were trained on 300 images over 1000 epochs. This result can be improved by training for a longer time over more images.

Lane Image from Car:
![image](https://user-images.githubusercontent.com/86836456/125584527-abf9bc4f-bbf1-4e12-8407-c2194737f475.png)

Masked Image:
![image](https://user-images.githubusercontent.com/86836456/125586355-99629f55-7243-4ce0-a0bf-097996eaec1c.png)


## Model Used

[Encoder Decoder Architecture](https://www.researchgate.net/publication/328460117_An_efficient_encoder-decoder_CNN_architecture_for_reliable_multilane_detection_in_real_time) was used. It first convolves few layers and then deconvolves same number of layers to get same resolution of image with modification. A small change was made from the paper to the last layer to deconvolve only 1 channel instead of 3.

![image](https://user-images.githubusercontent.com/86836456/125587387-0f02209b-1fa6-4ee9-83d4-7b659698c8c1.png)


## Training Dataset

CARLA allows you to attach multiple sensors to your car from which you can draw data. One such sensor segments all objects from the image which includes lanes. I used this sensor to act as the output of my training data and the data from the rgb sensor as the input for the training data. Since CARLA is very close to real life situations and since you can model all types of weather in CARLA, it was a good choice to pull training data from CARLA.

To generate the input and output data, the RGB and the segmentation sensor was attached to the vehicle and the vehicle was set to automatic drive mode. The images were then saved every 1 second and stored in the respective folders. This way we can let the car run on any map for however long and we'll have a ton of input and output data.

Example Input Data:

![image](https://user-images.githubusercontent.com/86836456/125590483-9e0d9d36-9b1d-4825-9706-aff390af6103.png)

Example Output Data:

![image](https://user-images.githubusercontent.com/86836456/125590568-9c6b06b6-61b3-4d51-94dc-b565821f3200.png)

## Usage

### Dependencies
Run ``` pip install -r requirements.txt ```

### Generating Dataset
1. Run the CARLA Simulator
2. Next, to generate the input and output images run
   ``` python generate_ip_op.py ```.
   Run this for however long you desire to generate enough input and output images.
3. Once the input and output images are generated, we must complile all the input and output data into a pickle for easier reading. Run
   ``` python pickle_load.py ```

### Training
Set desired epochs (here it's set to 1000) and run ``` python lane_detection.py ```

### Testing on custom images
Change file name of the image you want to test on and also set the name of the trained model and run ``` python lane_test.py ```
