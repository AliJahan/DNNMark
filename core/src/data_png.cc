// The MIT License (MIT)
// 
// Copyright (c) 2016 Northeastern University
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in 
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "data_png.h"

namespace dnnmark {

std::unique_ptr<PseudoNumGenerator> PseudoNumGenerator::instance_ = nullptr;

static unsigned long long int seed = 1234;

PseudoNumGenerator::PseudoNumGenerator() {
    CURAND_CALL(curandCreateGenerator(&gen_, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen_, seed));
  }

  // PNG instance
PseudoNumGenerator::~PseudoNumGenerator() {
    CURAND_CALL(curandDestroyGenerator(gen_));
}

//static PseudoNumGenerator* PseudoNumGenerator::GetInstance() {
//}

void PseudoNumGenerator::GenerateUniformData(float *dev_ptr, int size) {
    CURAND_CALL(curandGenerateUniform(gen_, dev_ptr, size));
}

void PseudoNumGenerator::GenerateUniformData(double *dev_ptr, int size) {
    CURAND_CALL(curandGenerateUniformDouble(gen_, dev_ptr, size));
}  

//std::unique_ptr<PseudoNumGenerator> PseudoNumGenerator::instance_ = nullptr;

} // namespace dnnmark

