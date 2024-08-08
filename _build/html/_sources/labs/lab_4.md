# Lab 4 : Elementwise Unit

## Goal of this lab
---
- [Porting and Profiling the Models - 20%](#porting-and-profiling-the-models---20)
- [Accelerating the Logistic Function - 60%](#accelerating-the-logistic-function---60)
- [Accelerating the Softmax Function - 20%]()

## Introduction
---
Modern models frequently utilize specialized activation functions that involve complex mathematical computations, such as exponentials, square roots, and reciprocals. Unlike simpler functions like ReLU, these sophisticated operations can become bottlenecks during model inference. In this lab, we will design an element-wise unit specifically for enhancing the inference speed of the **Logistic** and **Softmax** functions in the **MobileViT** model.

## Porting and Profiling the Models - 20%
---
### Porting the New Models - 10%

In this lab, we will provide you two models:

1. Logistic Test Model

    This model serves as a benchmark to verify the correctness of your design. In other words, your design should pass the golden test of this model in order to get the score.

2. MobileViT

    This is the actual model we aim to accelerate. We will use this model to benchmark the performance of your design.

> [model download link]()

After downloading and unzipping the files, place the two model folders directly into the `CFU-Playground/common/src/models` directory.

Just like we did in lab 1, we should modify some files to add the new models: 

`CFU-Playground/common/src/models/models.c`
```c
#if defined(INCLUDE_MODEL_DS_CNN_STREAM_FE)
        MENU_ITEM(AUTO_INC_CHAR, "Ds cnn stream fe", ds_cnn_stream_fe_menu),
#endif
// add cods below
#if defined(INCLUDE_MODEL_MOBILE_VIT_XXS)
        MENU_ITEM(AUTO_INC_CHAR, "MobileViT xxs", mobileViT_xxs_menu),
#endif
#if defined(INCLUDE_MODEL_LOGISTIC)
        MENU_ITEM(AUTO_INC_CHAR, "Logistic test", quant_logistic_model_menu),
#endif
```

`CFU-Playground/common/src/tflite.cc`
```cpp
#ifdef INCLUDE_MODEL_DS_CNN_STREAM_FE
    2048 * 1024,
#endif
// add codes below
#ifdef INCLUDE_MODEL_MOBILE_VIT_XXS
    16384 * 1024,
#endif
#ifdef INCLUDE_MODEL_LOGISTIC
    50 * 1024,
#endif
```

`CFU-Playground/proj/<lab4 proj folder>/Makefile`
```sh
DEFINES += INCLUDE_MODEL_MOBILE_VIT_XXS
DEFINES += INCLUDE_MODEL_LOGISTIC
```

Also, please follow this guide to **add the new operations required by the MobileViT**.
> [Add ops in CFU_Playground](https://hackmd.io/@Tsai-Wooo/SkjTbXdJ0)

After completing these steps, you should be able to run the two new models and obtain the scores for this section.

### Profiling the MobileViT - 10%

After executing the MobileViT model, you may notice that the inference process is quite slow. Please **analyze and compare the time consumed by each operation**, with particular attention to the **activation functions**. You may present your analysis in any format, such as **a table or a pie chart**. Additionally, you have the option to use data with or without SIMD MAC acceleration.

Here is an example of a pie chart:  
<img src="images/lab4/pie_example.png" width="1000px">

```{hint}
You can use either the perf counter or the ticks displayed in the results after execution as your data source.  
<img src="images/lab2/conv_result.png" width="300px">
```

Next, trace the code of the Logistic function to identify the **complex mathematical computations involved**, which may be contributing to the slow execution.

Make sure to present your findings to the TAs during the demo.
```{hint}
You can start tracing the code from this file, and the model will use the topmost overloaded `Logistic(...)` function:  
`CFU-Playground/third_party/tflite-micro/tensorflow/lite/kernels/internal/reference/integer_ops/logistic.h` Then, you will find the mathematical computations are defined in:  
`CFU-Playground/third_party/tflite-micro/third_party/gemmlowp/fixedpoint/fixedpoint.h`
```

## Accelerating the Logistic Function - 60%
---

\begin{gather*}
\text{logistic}(x) = \frac{1}{1 + e^{-x}}
\end{gather*}

Add the integer version of the Logistic function to your project.
```sh
$ cp \
  ../../third_party/tflite-micro/tensorflow/lite/kernels/internal/reference/integer_ops/logistic.h \
  src/tensorflow/lite/kernels/internal/reference/integer_ops/logistic.h
```
````{important}
Add the **sixth perf counter** at the beginning and end of the topmost Logistic function within the `logistic.h` file in your project. **This step is crucial for evaluating your score, so please ensure it is not overlooked.**

```cpp
inline void Logistic(int32_t input_zero_point, int32_t input_range_radius,
                     int32_t input_multiplier, int32_t input_left_shift,
                     int32_t input_size, const int8_t* input_data,
                     int8_t* output_data) {
  // Integer bits must be in sync with Prepare() function.
  perf_enable_counter(6);

  ...

  perf_disable_counter(6);
}
```
````

### Evaluation Criteria

```{attention} 
You will get **0%** if you can't pass the golden test of the Logistic Test Model.
```

TBD

### Guide

```{note}
**You are not required to follow to the provided guide below**. Instead, you are encouraged to use any method to accelerate model inference, provided that it passes the golden test of the Logistic Test Model.
```
First of all, it is essential to familiarize yourself with **fixed-point** arithmetic and the `FixedPoint` class defined in `fixedpoint.h`. The key components and functions you are likely to use within the `FixedPoint` class include `kIntegerBits`, `FromRaw()`, and `raw()`. Additionally, you may find the comments for the `FixedPoint` class in `fixedpoint.h` to be a useful reference.

After tracing the code, you may notice that the **exponential** and the **reciprocal** are the primary bottlenecks, so we can focus on accelerating these two operations in this function.  
`third_party/gemmlowp/fixedpoint/fixedpoint.h`
```cpp
// Returns logistic(x) = 1 / (1 + exp(-x)) for x > 0.
template <typename tRawType, int tIntegerBits>
FixedPoint<tRawType, 0> logistic_on_positive_values(
    FixedPoint<tRawType, tIntegerBits> a) {
  return one_over_one_plus_x_for_x_in_0_1(exp_on_negative_values(-a));
}
```

#### Software

Below is an example of replacing the original software-computed exponential with the CFU operation:
```cpp
template <typename tRawType, int tIntegerBits>
FixedPoint<tRawType, 0> exp_on_negative_values(
    FixedPoint<tRawType, tIntegerBits> a) {
  typedef FixedPoint<tRawType, 0> ResultF;
  ...
  b = cfu_op0(0, 0, 0);
  return b;
}
```
This is just a simple example. You should properly design your CFU op to pass the data needed for the hardware unit. Also, you should properly handle the conversion of `FixedPoint` and the `tRawType` using `FromRaw()` and `raw()`.

```{note}
Given the symmetry of the Logistic function, it's only necessary to consider either positive or negative inputs for the exponential computation, with the negative ones being simpler to manage.
```

#### Hardware
As for the hardware unit section, you should first familiarize yourself with the CFU handshake signals.
> [Details and Use Cases of the CPU <-> CFU interface](https://cfu-playground.readthedocs.io/en/latest/interface.html)

Here is a dummy example:
```verilog
assign cmd_ready = ~rsp_valid & ~calculating;
always @(posedge clk) begin
    if (reset) begin
        ...
    end else if (rsp_valid) begin
        rsp_valid <= ~rsp_ready;
    end else if (cmd_valid) begin
        if (...) begin
            ...
            input_valid <= 1;
            calculating <= 1;
        end
        ...
    end
    else if (input_valid) begin
        input_valid <= 0;
    end
    else if (finish == 1) begin
        calculating <= 0;
        rsp_valid <= 1'b1;
        rsp_payload_outputs_0 <= result;
    end
end
```

For calculating results using hardware, you have the option to employ either a **Lookup Table** or Mathematical Approximation methods, such as the **Taylor Series**, **Newton-Raphson division**, or **Polynomial Approximation**.

In this lab, for the **exponential function, we recommend using either a Lookup Table or the Taylor Series**. For the **reciprocal function, we suggest using a Lookup Table or the Newton-Raphson division method**, similar to the approach used in one_over_one_plus_x_for_x_in_0_1(), but implemented in hardware.

## Accelerating the Softmax Function - 60%
`tensorflow/lite/kernels/internal/reference/softmax.h`

\begin{gather*}
\text{softmax}(x) = 
\end{gather*}

Add the integer version of the Softmax function to your project.

````{important}
Add the **fifth perf counter** at the beginning and end of the **second** Softmax function within the `softmax.h` file in your project. **This step is crucial for evaluating your score, so please ensure it is not overlooked.**

```cpp

```
````