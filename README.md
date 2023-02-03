# Examination of Several Neural Networks for Facial Emotion Recognition
this project is a bachelor thesis in  **TU Bergakademie Freiberg**. In this project, different CNN models for emotion recognition based on facial features are investigated..


# Tools
* [Tesnorflow](https://www.tensorflow.org/)  
  * ``` pip install tensorflow ``` 
* [OpenCv](https://docs.opencv.org/3.4/index.html)
  * ``` pip install opencv-python ``` 
* [dataset:  FER-2013](https://www.kaggle.com/datasets/msambare/fer2013?select=train)


# Run App
* help: `python3 main.py -h`

      -t TEST, --test TEST      test model
      -l TRAIN, --train TRAIN   train model
---
* to testing by webcam
  * with first Model:
    * strategy 1: `python3 main.py -t M1S1`
    * strategy 2: `python3 main.py -t M1S2`
    * strategy 3: `python3 main.py -t M1S3`
  * with second Model:
    * strategy 1: `python3 main.py -t M2S1`
    * strategy 2: `python3 main.py -t M2S2`
    * strategy 3: `python3 main.py -t M2S3`
  * with third Model:
    * strategy 1: `python3 main.py -t M3S1`
    * strategy 2: `python3 main.py -t M3S2`
    * strategy 3: `python3 main.py -t M3S3`
---
* to testing by image:  `python3 main.py -i "image_path"`
  * with first Model:
    * strategy 1: `python3 main.py -i "image_path" M1S1`
    * strategy 2: `python3 main.py -i "image_path" M1S2`
    * strategy 3: `python3 main.py -i "image_path" M1S3`
  * with second Model:
    * strategy 1: `python3 main.py -i "image_path" M2S1`
    * strategy 2: `python3 main.py -i "image_path" M2S2`
    * strategy 3: `python3 main.py -i "image_path" M2S3`
  * with third Model:
    * strategy 1: `python3 main.py -i "image_path" M3S1`
    * strategy 2: `python3 main.py -i "image_path" M3S2`
    * strategy 3: `python3 main.py -i "image_path" M3S3`
---
* to training:
  * first model:
    * strategy 1: `python3 main.py -l model_1 strategy_1 `
    * strategy 2: `python3 main.py -l model_1 strategy_2 `
    * strategy 2: `python3 main.py -l model_1 strategy_3 `

  * second model:
    * strategy 1: `python3 main.py -l model_2 strategy_1 `
    * strategy 2: `python3 main.py -l model_2 strategy_2 `
    * strategy 3: `python3 main.py -l model_2 strategy_3 `

  * third model:
    * strategy 1: `python3 main.py -l model_3 strategy_1`
    * strategy 2: `python3 main.py -l model_3 strategy_2`
    * strategy 3: `python3 main.py -l model_3 strategy_3`
---
* tensorBoard: after or while training run following commands in another tab

      rm -rf ./logs/ # or change the name 
      tensorboard --logdir logs/fit

* clean: ``` py3clean . ```

# Comparsion
* to compare between viola&jones methode and using of dlib to recognize face
        
      cd Recognition/compare
      python3 compare.py -c compare
* to test the functionality of viola&jones to recognize eyes and mouth in an image
  
      cd Recognition/compare
      python3 compare.py -i "image_path"

[//]: # (# Implementation)

[//]: # (* just call [*saveImageFromCamera&#40;&#41;*]&#40;Recognition/face/camera.py&#41; in *main.py* from class *Camera*)

[//]: # (* [Load dataset]&#40;Recognition/Emotion/help_functions.py&#41;)

[//]: # (  * **Train**: by calling *load_dataset&#40;"Recognition/archive/train/"&#41;* )

[//]: # (  * **Test**: by calling *load_dataset&#40;"Recognition/archive/test/"&#41;*)

# Preprocessing
* to detect face and to build a black mask around it
  ```
    cd Recognition
    python3 preprocessing.py -f face
  ```

* to detect mouth & eyes and to build a black mask around them
  ```
    cd Recognition
    python3 preprocessing.py -em eyeMouth
  ```

# Screenshots
## result
* Angry:
![](pictures_for_test/with_Emotion/Screenshot_An.png)
* Fear:
![](pictures_for_test/with_Emotion/Screenshot%20from%202022-12-12%2012-12-44.png)
* Disgusted:
![](pictures_for_test/with_Emotion/Screenshot%20from%202022-12-12%2012-13-09.png)
* Neutral:
![](pictures_for_test/with_Emotion/Screenshot%20from%202022-12-12%2012-14-19.png)
* Happy:
![](pictures_for_test/with_Emotion/Screenshot_Ha.png)
* Surptised:
![](pictures_for_test/with_Emotion/Screenshot_Su%20.png)

    
# uninstall package
```
pip uninstall "package name"

```
