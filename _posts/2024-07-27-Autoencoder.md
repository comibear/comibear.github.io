---
title: "[AI] Intro to Autoencoder"
description: "Simple approach to Autoencoder"
writer: Sangyun Won
categories:
  - AI
tags:
  - AI

image:
  lqip: data:image/webp;base64,UklGRpoAAABXRUJQVlA4WAoAAAAQAAAADwAABwAAQUxQSDIAAAARL0AmbZurmr57yyIiqE8oiG0bejIYEQTgqiDA9vqnsUSI6H+oAERp2HZ65qP/VIAWAFZQOCBCAAAA8AEAnQEqEAAIAAVAfCWkAALp8sF8rgRgAP7o9FDvMCkMde9PK7euH5M1m6VWoDXf2FkP3BqV0ZYbO6NA/VFIAAAA
  alt: Responsive rendering of Chirpy theme on multiple devices.

toc: true
toc_sticky: true

date: 2024-07-27
last_modified_at: 2024-07-27
---

# 1. What is Autoencoder?

Autoencoder은 한마디로 데이터를 더욱 작은 차원으로 압축해주는 모델이다. 이를 처음 접해보는 사람이라면, 데이터를 더 작은 차원으로 압축하는 모델이 가지는 의의가 뭐지? 라는 생각이 들 수도 있을 것이다. 하지만 AI 가 점점 더 다양한 분야에 적용됨에 따라서, 데이터셋의 크기가 매우 큰 경우 또한 많이 존재하게 되고, 더욱 작은 차원(크기)로 큰 데이터를 표현할 수 있다면, 큰 데이터들을 매우 효율적으로 학습시킬 수 있다.

가장 중요한 것은 Input data 의 특징을 가장 효율적으로 압축하는 것이며, 이 정도를 객관적으로 파악하기 위한 방식이 바로 원래 데이터로 복구해보는 것읻다. 이 줄여진 데이터를 다시 원래 데이터로 복구할 수 있어야 기존의 데이터의 특징을 가장 잘 담고 있다고 할 수 있다. 

![image](https://github.com/user-attachments/assets/5438ccdc-ec77-40a5-9386-170d47158f3c)

추가로, 대부분의 AI 는 일종의 블랙박스 모델이다. 내부의 구조에 따라서 "어떻게" 특징들이 정의되는지, 이 특징들을 어떻게 학습하는지는 관심이 없다. 그냥 작은 차원으로 데이터를 압축했더니, 각각의 파라미터들이 기존 데이터의 특징을 나타내는 값을 가지고 있다 ~ 라고 살짝은 무책임하게 생각해야 한다. 

# 2. Model Structure

Autoencoder의 구조는 input data 의 shape 에 따라서 유동적이다. 이 포스트에서는 MNIST 의 손글씨 데이터를 기준으로 Model 을 구현할 것이기 때문에, 이 데이터들에 대한 Autoencoder의 Model Structure 을 설명한다.

![image](https://github.com/user-attachments/assets/799ec9fd-2b16-4bec-a00f-baa0348bd0d0)

기본적인 구조는 위와 같다. input data를 Encoder 을 통해서 latent vector (code) 로 압축하고, 이 latent vector 을 다시 Decoder 을 이용해서 원본 데이터로 복구하게 된다.

MNIST dataset 은 