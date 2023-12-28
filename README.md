작성일시 : 2020.03.08

---
**가상 화장 프로젝트.**

총 진행 기간 2018.10 - 2019.02

립제품에 한하여 화장 전 얼굴과 화장품에 대한 정보를 바탕으로 화장 후의 결과를 예측해보는 프로젝트.

제공되는 어플에서 실행될 서비스에 대한 가능성을 보기위한 프로젝트.

---
**DATASET**

총 12명의 사람, 3종류의 빛(조명)의 세기, 2종류의 화장품을 바른 강도(정도), 12개의 화장품에 대하여 dataset을 취득후 진행.

빛의 세기 - 150lux, 250lux, 1000lux

화장품을 바른 강도 - 중, 강

립스틱 색(color space를 rgb, hsv, lab 뿐만 아니라 변형시킨 xyz로도 사용함.)

![colors](https://github.com/JooJiyun/virtual_makeup/blob/master/Assets/lipstick_colors.png)

---
**진행 방법**

화장 전후의 입술 색에 대한 정보는 각각 입술 영역을 landmark로 추출하여 평균 색을 사용.

(1) 제공되는 어플의 파라미터를 추정하는 방법. 

휴대폰 어플상에서 화장후의 색이 보이게 하는 optimal parameter 찾은 후 그 parameter를 추정. 

각각의 optimal parameter는 화장후의 색 정보에서 선형 변환을 한 것으로 사용.

물론 optimal parameter를 찾는 선형회귀의 parameter를 계속 바꾸어 optimal parametr 또한 계속 바꿔가면서 실험.

![way1](https://github.com/JooJiyun/virtual_makeup/blob/master/Assets/lipsticks_1.png)

(2) 프로젝트의 원래의 취지와는 별개로 제공되는 어플을 고려하지 않고 바로 화장 후의 색을 추정.

![way2](https://github.com/JooJiyun/virtual_makeup/blob/master/Assets/lipsticks_2.png) 

결과비교는 LAB color space에서 진행. 


---

1) linear regression, svm, NN

사람 얼굴에서 입술 부분만 landmark로 떼어내어 화장 전후에 대하여 색 정보를 얻어내어
화장전의 입술 색과 화장품의 색, 강도, 빛의 세기를 인풋으로 하여 화장후의 색을 아웃풋으로 내도록 하였다.

각각 RGB, HSV, XYZ, LAB color space에서 진행해보았으나 결과는 그리 좋지 않았다.

color space의 차이가 좋은 결과를 낼 수도 있겠지만 결국엔 각각의 color space는 서로 선형변환을 한 것이기 때문에 큰 차이가 나지 않았다고 생각한다.


---
2) CNN

인풋에 대한 정보를 색 정보만 사용하지 않고 이미지를 그대로 사용해 보기 위해 시도.

입술에 대한 landmark를 추출한 이미지 외에도 입술영역을 모두 포함할만큼만으로 crop시킨 이미지를 사용하는 등 인풋 이미지에 변형을 가해 시도함.

립스틱에 대한 정보 또한 fully connected layer에서 값으로 추가하는 것 외로도 위의 dataset에 있는 립스틱 색 이미지와 같이 이미지로서도 넣어봄.

layer를 몇개만 사용하는 간단한 네트워크부터 시작하여, vgg, Unet등을 간단하게 변형시켜 시도.

Image Style Transfer Using Convolutional Neural Networks (2016 IEEE)
: CNN을 사용하여 그림의 스타일을 변형시킴. 19layer의 vgg를 base로 함.

PairedCycleGAN: Asymmetric Style Transfer for Applying and Removing Makeup (2018 CVPR)
: U-net과 STN 대신 generator로 U-net에 skip connection을 연결하여 Dilated ResNet (DRN) 아키텍처를 사용. 
generator의 구조를 참고하여 변형된 dilated resnet을 사용.


참고한 논문등을 바탕으로 feature map의 크기들, dropout, 연산 종류, activation function, loss term등을 바꿔가며 실험.

하지만 epoch을 크게 할 수록 결과가 립스틱 색상 계열에 따라 치우친 결과가 나오게 됨.

![cnn result](https://github.com/JooJiyun/virtual_makeup/blob/master/Assets/cnn_result.png) 

수치적으로도 별로 결과가 그리 좋지 않았음.

![cnn loss](https://github.com/JooJiyun/virtual_makeup/blob/master/Assets/CNN_loss.png)

LAB에서 차이가 3이내여야 괜찮은건데 아래 그래프만 봐도 알 수 있듯이 대부분이 3을 넘었다.

resource등을 이유로 네트워크를 깊게 만들지 못했지만 네트워크만 깊어진다고 해결될 것처럼 보이지 않았다.

그래서 내 컴퓨터의 gpu가 좋아지게 되어서도 이어서 CNN의 네트워크를 개발시키지 않고 GAN을 사용해 보기로 하였다.


---
3) GAN

내 컴퓨터의 gpu가 좋아지면서 가능하게 된 실험.

아웃풋의 값을 바로 추정하지말고 이미지를 아웃풋으로 해보기로 한 시도.

앞서 참고했던 PairedCycleGAN: Asymmetric Style Transfer for Applying and Removing Makeup의 구조를 다시 참고한 cycle gan으로 실험.

논문과 마찬가지로 makeupGAN과 removeGAN으로 이루어진 네트워크지만 paired하지는 않다.

![not used](https://github.com/JooJiyun/virtual_makeup/blob/master/Assets/not_used.png)

본래의 논문대로라면, 위의 그림과 같이 구성하여 data를 더 모아 실험이 가능 했겠지만 아래와 같은 이유들(unpaired data)로 인해 구조를 바꾸게 되었다.


다른 사람이라도 같은 립스틱 목록을 가지고 있기 때문에 본래의 cycle GAN처럼 데이터를 더 모아서 학습시킬 수 없다.


예를 들어 fake B와 fake B’이 같은 사람일 경우에는 립스틱이 무조건 다른 경우들이므로 학습할 필요가 없거나 학습할 수가 없다.

fake B와 fake B’이 다른 사람일 경우, 립스틱의 번호와 무관하게 같은 립스틱 목록에 대하여 makeup 이미지가 있으므로 학습할 필요가 없다.

바꾼 구조는 위처럼 generator를 통과한 fake 데이터들을 pool에 넣어 data를 모으지 않고 

아래와 같이 바로 cycle data를 만들도록 하여 cycle data만 pool에 넣을 수 있도록 하였다.

![used](https://github.com/JooJiyun/virtual_makeup/blob/master/Assets/used.png)

generator 구조들:

![generator](https://github.com/JooJiyun/virtual_makeup/blob/master/Assets/generator.png)

CNN때와 마찬가지로 네트워크는 조금 간소화 시키고 여러가지를 변경시키면서 실험하였다.

loss는 generator loss, descriminator loss, cycle consistency loss에 각각 상수배하였고 각각의 상수는 실험을 통하여 조정하였다.

결과:

![gan result](https://github.com/JooJiyun/virtual_makeup/blob/master/Assets/gan_result.png)

제일 나은 결과를 얻을 수는 있었지만 그리 좋은 결과는 아니었고 blending과 같은 후처리까지 고려해보았지만 사용하기엔 어려웠다.


