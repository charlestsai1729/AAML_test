# Lab 2 Quantization and SIMD MAC

## Introduction

In the previous lab, we successfully ran the float32 Keyword Spotting (KWS) model on CFU-Playground. In this lab, we will focus on running a quantized int8 KWS model and leverage the benefits of quantization to achieve acceleration.

## Run Quantized Model - 10%

```sh
$ cd CFU-Playground/proj
$ cp -r <lab1 proj folder> <lab2 proj folder>
$ cd <lab2 proj folder>
```

Quantizing float model to int8 is kinda complicated and is not the focal point of this lab, so we have provided the quatized model for you.
