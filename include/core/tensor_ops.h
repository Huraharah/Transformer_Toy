#pragma once

#include "core/tensor.h"

void tensorFillCUDA(Tensor& tensor, float value);
void tensorScaleCUDA(Tensor& tensor, float scalar);
void tensorAddCUDA(const Tensor& a, const Tensor& b, Tensor& out);