---
title: "[AI] Intro to Autoencoder"
description: "Simple approach to Autoencoder"
writer: Sangyun Won
categories:
  - AI
tags:
  - AI

image:
  path: https://github.com/user-attachments/assets/0af299a1-c95d-4720-b6dc-43888b8f5af7
  alt: Responsive rendering of Chirpy theme on multiple devices.

toc: true
toc_sticky: true

date: 2024-07-27
last_modified_at: 2024-07-27
---

# 1. What is Autoencoder?

`Autoencoder`은 한마디로 데이터를 더욱 작은 차원으로 압축해주는 모델이다. 이를 처음 접해보는 사람이라면, 데이터를 더 작은 차원으로 압축하는 모델이 가지는 의의가 뭐지? 라는 생각이 들 수도 있을 것이다. 하지만 AI 가 점점 더 다양한 분야에 적용됨에 따라서, 데이터셋의 크기가 매우 큰 경우 또한 많이 존재하게 되고, 더욱 작은 차원(크기)로 큰 데이터를 표현할 수 있다면, 큰 데이터들을 매우 효율적으로 학습시킬 수 있다.

가장 중요한 것은 Input data 의 특징을 가장 효율적으로 압축하는 것이며, 이 정도를 객관적으로 파악하기 위한 방식이 바로 원래 데이터로 복구해보는 것읻다. 이 줄여진 데이터를 다시 원래 데이터로 복구할 수 있어야 기존의 데이터의 특징을 가장 잘 담고 있다고 할 수 있다. 

![image](https://github.com/user-attachments/assets/5438ccdc-ec77-40a5-9386-170d47158f3c)

추가로, 대부분의 AI 는 일종의 블랙박스 모델이다. 내부의 구조에 따라서 "어떻게" 특징들이 정의되는지, 이 특징들을 어떻게 학습하는지는 관심이 없다. 그냥 작은 차원으로 데이터를 압축했더니, 각각의 파라미터들이 기존 데이터의 특징을 나타내는 값을 가지고 있다 ~ 라고 살짝은 무책임하게 생각해야 한다. 

# 2. Discrete vs Continuous

앞서 이야기했던 `Autoencoder`은 `Discrete` 하다. 예를 들어서 MNIST 손글씨 데이터를 `Autoencoder`를 이용해 `latent space`로 매핑하는 경우를 생각해보자. 각각의 이미지는 `latent space` 로 압축되며, 같은 label (숫자) 끼리는 서로 가깝게 유지되는, 일종의 `Cluster` 을 이루게 된다. 

그렇다면 과연 이 `Cluster` 의 범위를 잡아 임의의 점을 정의하고, 이 점을 `Decoder` 에 올리게 되면 새로운 이미지가 생성되는 것이 아닐까? (일종의 `Generative Model`을 생각할 수 있지 않을까?) 안타깝게도 불가능하다. 정확히 말하면 정당성이 없다고 할 수 있다. 

![image](https://github.com/user-attachments/assets/45bd68be-1c2f-41f6-88c8-ed0ac6989f36)*Autoencoder vs Variational Autoeencoder, Don't need to know now*

이는 제목에서 보다시피 `Autoencoder`가 각각의 Input data들을 `Discrete`한 `latent space`로 매핑하기 때문에, 정말 각각의 이미지 데이터에 대한 정보밖에 담고 있지 않다. 이외의 점들에 대해서는 의미가 존재하지 않는다. (물론 직관적으로 `Cluster`가 생성되었기에, 유사한 특징들을 내포하고 있어 어느 정도의 유사성을 확보되긴 한다.)

따라서 아래의 사진을 보다시피, `Generative Model` 로 사용하기 적합하지 않다. 

![image](https://github.com/user-attachments/assets/561ace2e-566c-455a-9c0b-eefc1ee32500)*Generated Images using means of cluster in latent space*

이를 보완하기 위해, 일종의 확률분포를 도입하여, Variational Autoencoder이라는 새로운 모델이 제시되었다. 

# 3. Variational Autoencoder

Variational Autoencoder 은 기존의 Autoencoder에 확률을 도입한다. 기존에는 Discrete하게 하나의 이미지에 하나의 latent vector만을 생성해냈다면, Variational Autoencdoer 은 latent space 에서 하나의 점을 중심으로 [Gaussian distribution](https://en.wikipedia.org/wiki/Normal_distribution) 을 형성한다. 

<br>

![image](https://github.com/user-attachments/assets/a98f1842-646c-438b-84e1-9be00013eb22)*Image from https://medium.com/@hugmanskj/autoencoder-%EC%99%80-variational-autoencoder%EC%9D%98-%EC%A7%81%EA%B4%80%EC%A0%81%EC%9D%B8-%EC%9D%B4%ED%95%B4-171b3968f20b*

<br>

`Gaussian Distirbution`은 평균과 분산을 통해서 정의되기 때문에, `VAE`의 `Encoder` 부분은 `Latent space`의 `dimension`에 따라 **mean**과 **diversion**을 학습하여, 그에 따른 확률분포를 생성해낸다. (사실상 mean부분과 diversion부분을 우리가 정의하고, 개념적으로만 생성한다는 표현이 옳다.)

<br>
<br>

![image](https://github.com/user-attachments/assets/c1c79826-de04-4685-beeb-68169cdda7a5)*Image from https://medium.com/@hugmanskj/autoencoder-%EC%99%80-variational-autoencoder%EC%9D%98-%EC%A7%81%EA%B4%80%EC%A0%81%EC%9D%B8-%EC%9D%B4%ED%95%B4-171b3968f20b*

그런데 한 가지 문제가 있다. 이렇게 형성된 `Gaussian Distribution은` 하나의 확률분포이기 때문에, `Decoder`의 Input이 될 수 없다. `Decoder`의 Input으로는 하나의 `latent vector`이 정의되어야 하는데, 이를 어떻게 해소할까?

바로 `Sampling` 기법이다. 사실 이름만 거창하지, 기존에 만들어진 확률분포를 기준으로 임의의 벡터를 생성하는 것이다. 당연하게도 각 `Gaussian Distribution`의 확률에 따라서 `Sampling` 이라는 이름으로 벡터가 정의된다. 

이를 식으로 나타내면 다음과 같다. 

$$x_{sampled} = \mu + \sigma \times \epsilon$$  
$$\mu = mean,\; \sigma = diversion,\; \epsilon = random$$


얼핏 보면 매우 단순하지만, 이 식에는 **`reparamerterize trick`** 이 적용되었다. 기존의 $\sigma$ 를 통해서 곧바로 `noise` 를 적용시켜주는 것이 아니라, 새로운 파라미터인 $\epsilon$ 을 만들어서, 곱해주는 방식을 사용해주는 것이다. 만약에 $\sigma$ 를 직접 사용하게 되면, $\sigma$에 대한 역전파 계산이 불가능해지기 때문에, 랜덤성을 가지는 새로운 변수를 만들어 곱해주는 방식을 사용한다. 

# 4. More about Variational Autoencoder

사실, Variational Autoencoder 과 기존의 Autoencoder 은 아예 다른 개념으로부터 파생되었다고 할 수 있다. 생긴 것은 매우 유사하지만, Autoencoder 은 encoder 을 만들어내기 위해서 decoder 을 이어붙은 모델이고, Variational Autoencoder 은 decoder 을 만들기 위하여 encoder 이라는 부분을 더해준 것이다. 

 
 그렇기에 Variational Autoencoder은 하나의 Generative Model로써 사용이 가능하다. latent space 로 매핑된 평균과 분산의 변수들을 활용해서 sampling 하게 되면, 이를 decode 했을 시에 유의미한 새로운 이미지가 탄생한다. 

![image](https://github.com/user-attachments/assets/de1754d6-8a26-4b5e-bb10-8f1f9846709d)    

이는 앞서 보였던 Autoencoder의 방식과 유사하게, latent space에서 각 벡터를 샘플링하여 디코딩한 이미지인데, Autoencoder 에 비하여 매우 높은 이미지 생성 능력을 가진 것을 볼 수 있다. 

