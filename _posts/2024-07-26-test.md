---
title: "[AI] Intro to Autoencoder"
description: "Simple approach to Autoencoder"
writer: Sangyun Won
categories:
  - AI
tags:
  - AI

toc: true
toc_sticky: true

date: 2024-07-27
last_modified_at: 2024-07-27
---

# 1. What is Autoencoder?
Autoencoder은 한마디로 데이터를 더욱 작은 차원으로 압축해주는 모델이다. 이를 처음 접해보는 사람이라면, 데이터를 더 작은 차원으로 압축해서 뭐해? 도대체 이 모델이 가지는 의의가 뭐지? 라는 생각이 들 수도 있을 것이다. 하지만 AI 가 점점 더 다양한 분야에 적용됨에 따라서, 데이터셋의 크기가 매우 큰 경우 또한 많이 존재하게 되고, 더욱 작은 차원(크기)로 큰 데이터를 표현할 수 있다면, 큰 데이터들을 매우 효율적으로 학습시킬 수 있을 것이다. 

한 가지 예를 들자면, MNIST 의 손글씨 데이터를 생각해볼 수 있다. 이 경우 기존의 데이터는 1x28x28 크기의 데이터지만, 우리는 그냥 0, 1, 2, 3 ... 9 로 바로 표현할 수도 있겠다. (물론 데이터를 완전히 표현하지는 못하겠지만) 이렇게 작은 차원으로 줄인 데이터들은 

![image](https://github.com/user-attachments/assets/17500308-1cab-4903-9b5c-e4a38b14710f)
