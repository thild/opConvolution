// 
// main.cpp
//  
// Author:
//       Tony Alexander Hild <tony_hild@yahoo.com>
// 
// Copyright (c) 2011 
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
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.



#include <iostream>
#include <stdio.h>
#include "timer.h"
#include "statistics.h"
#include "util.h"
#include <assert.h>
#include <list>

//#include <hwloc.h>
//#include <callgrind.h>
#include <fstream>
#include <math.h>

#include "opConvolutionFilter.h"

//#include <xmmintrin.h>  // SSE  (Required to use the __m128, and __m128d type)
//#include <emmintrin.h>  // SSE2 (Required to use the __m128i type)
//#include <pmmintrin.h>  // SSE3
// 
//#ifdef __SSE4_1__
//#include <smmintrin.h>  
//#endif 

                         
using namespace std;
timer m_Timer; 

void loopBlockAlignedSSEConvolveTest();

//
//
//void run2DTest(const string testName, const int iterations, list<int>& kernels, const int imageStride, const int imageWidth, const int imageHeight, 
//             const int kernelStride, const int kernelWidth, 
//             const float* inputImage, float* outputImage, const float* kernel);

static bool assertConvolution(const float* controlOutput, const float* convolveOutput,
                           int imageWidth, int imageHeight, int controlStride, 
                           int convolveStride, int kernelWidth);
                                  
//inline static float convolution(const float *image, int stride, const float *kernel, int kernelWidth, int x, int y);
float* gaussianKernel2D(const int kernelWidth, const float sigma);
float* gaussianKernel1D(const int kernelWidth, const float sigma);


ostream& tab( ostream& output ) { return output << '\t'; } 


void run2DTest(void (*convolutionFunction)(const int imageStride, const int imageWidth, const int imageHeight, 
                                         const int kernelStride, const int kernelWidth, 
                                         const float* inputImage, float* outputImage, const float* kernel), 
               const string testName, const int iterations, list<int>& kernels,  
               const int minKernelWidth, const int maxKernelWidth,
               const int imageStride, const int imageWidth, const int imageHeight, 
               const float* inputImage, float* outputImage) { 
              
    cout.setf(ios::fixed);
    cout.precision(3);  
    
     //#if aligned
    cout << left << setw(40) << testName;
    for ( list<int>::iterator it = kernels.begin(); it != kernels.end(); it++ ) {
        int kernelWidth = *it; 
        if(kernelWidth >= minKernelWidth && (kernelWidth <= maxKernelWidth || maxKernelWidth == 0)) {
            int kernelStride = calculateAlignedStride(kernelWidth, sizeof(float), ALIGMENT_BYTES);
            float* kernel = gaussianKernel2D(kernelWidth, 2);
            clear2DBuffer(outputImage, imageStride, imageHeight);
            float m = 0;
            vector<float> iter;
            for (int i = 0; i < iterations; i++) {
                m_Timer.start();    
                convolutionFunction(imageStride, imageWidth, imageHeight, 
                             kernelStride, kernelWidth, inputImage, 
                             outputImage, kernel);
                m_Timer.stop();
                iter.push_back(m_Timer.elapsed()); 
                m += m_Timer.elapsed();     
            }  
            delete[] kernel;
            m  = m / iterations;
            cout << setw(7) << m << setw(10) << StDev(iter) << flush; 
        }
        else {
            cout << setw(7) << "-" << setw(10) << "-" << flush; 
        }
        #ifdef DEBUG
        printImage(imageWidth, imageHeight, imageStride, outputImage);
        #endif
        
    }
    cout << endl;

} 
 


void runSSETest(const string testName, const int iterations, list<int>& kernels,  
               const int minKernelWidth, const int maxKernelWidth,
               const int imageStride, const int imageWidth, const int imageHeight, 
               const float* inputImage, float* outputImage) { 
              
    cout.setf(ios::fixed);
    cout.precision(3); 
              
     //#if aligned
    cout << left << setw(40) << testName;
    for ( list<int>::iterator it = kernels.begin(); it != kernels.end(); it++ ) {
        int kernelWidth = *it; 
        if(kernelWidth >= minKernelWidth && (kernelWidth <= maxKernelWidth || maxKernelWidth == 0)) {
            int kernelStride = calculateAlignedStride(kernelWidth, sizeof(float), ALIGMENT_BYTES);
            float* kernel = gaussianKernel2D(kernelWidth, 2);
            clear2DBuffer(outputImage, imageStride, imageHeight);
            float m = 0;
            vector<float> iter;
            for (int i = 0; i < iterations; i++) {
                m_Timer.start();    
                if(testName == "sse3Convolve"){
                    sse3Convolve(imageStride, imageWidth, imageHeight, 
                                 kernelStride, inputImage, 
                                 outputImage, kernel);
                }  
                else if(testName == "sse3CmConvolve"){ 
                    sse3CmConvolve(imageStride, imageWidth, imageHeight, 
                                 kernelStride, inputImage, 
                                 outputImage, kernel);                 
                }
                else if(testName == "sse3LbConvolve") {
                    sse3LbConvolve(imageStride, imageWidth, imageHeight, 
                                 kernelStride, inputImage, 
                                 outputImage, kernel);
                }
                else if(testName == "sse5Convolve") {
                    sse5Convolve(imageStride, imageWidth, imageHeight, 
                                 kernelStride, inputImage, 
                                 outputImage, kernel);
                }
                else if(testName == "sse7Convolve") {
                    sse7Convolve(imageStride, imageWidth, imageHeight, 
                                 kernelStride, inputImage, 
                                 outputImage, kernel);
                }
                else if(testName == "sse9Convolve") {
                    sse9Convolve(imageStride, imageWidth, imageHeight, 
                                 kernelStride, inputImage, 
                                 outputImage, kernel);
                }
                else if(testName == "sse11Convolve") {
                    sse11Convolve(imageStride, imageWidth, imageHeight, 
                                 kernelStride, inputImage, 
                                 outputImage, kernel);
                }
                m_Timer.stop();
                iter.push_back(m_Timer.elapsed()); 
                m += m_Timer.elapsed();     
            }  
            delete[] kernel;
            m  = m / iterations;
            cout << setw(7) << m << setw(10) << StDev(iter) << flush; 
        }
        else {
            cout << setw(7) << "-" << setw(10) << "-" << flush; 
        }
        #ifdef DEBUG
        printImage(imageWidth, imageHeight, imageStride, outputImage);
        #endif
        
    }
    cout << endl;

}

void runLoopBlockConvolveTest(const string testName, const int iterations, list<int>& kernels,  
               const int minKernelWidth, const int maxKernelWidth,
               const int imageStride, const int imageWidth, const int imageHeight, 
               const float* inputImage, float* outputImage, const int xBlock, const int yBlock) { 

    cout.setf(ios::fixed);
    cout.precision(3); 
              
     //#if aligned
    cout << left << setw(40) << testName;
    for ( list<int>::iterator it = kernels.begin(); it != kernels.end(); it++ ) {
        int kernelWidth = *it; 
        if(kernelWidth >= minKernelWidth && (kernelWidth <= maxKernelWidth || maxKernelWidth == 0)) {
            int kernelStride = calculateAlignedStride(kernelWidth, sizeof(float), ALIGMENT_BYTES);
            float* kernel = gaussianKernel2D(kernelWidth, 2);
            clear2DBuffer(outputImage, imageStride, imageHeight);
            float m = 0;
            vector<float> iter;
            for (int i = 0; i < iterations; i++) {
                if(testName == "loopBlock128x128Convolve") {
                    m_Timer.start();    
                    loopBlockConvolve (imageStride, imageWidth, imageHeight, 
                                  kernelStride, kernelWidth, inputImage, 
                                  outputImage, kernel, xBlock, yBlock);                     
                    m_Timer.stop();
                } 
                else if(testName == "loopBlockLoopUnroll16x64Convolve") {
                    m_Timer.start();    
                    loopBlockLoopUnrollConvolve (imageStride, imageWidth, imageHeight, 
                                  kernelStride, kernelWidth, inputImage, 
                                  outputImage, kernel, xBlock, yBlock);                     
                    m_Timer.stop();
                }
                else {
                    m_Timer.start();    
                    loopBlockAlignedSSEConvolve2 (imageStride, imageWidth, imageHeight, 
                                  kernelStride, kernelWidth, inputImage, 
                                  outputImage, kernel, xBlock, yBlock);
                     
                    m_Timer.stop();
                }
                iter.push_back(m_Timer.elapsed()); 
                m += m_Timer.elapsed();     
            }  
            delete[] kernel;
            m  = m / iterations;
            cout << setw(7) << m << setw(10) << StDev(iter) << flush; 
        }
        else {
            cout << setw(7) << "-" << setw(10) << "-" << flush; 
        }
        #ifdef DEBUG
        printImage(imageWidth, imageHeight, imageStride, outputImage);
        #endif
        
    }
    cout << endl;
}


void runScTest(const string testName, const int iterations, list<int>& kernels,  
               const int minKernelWidth, const int maxKernelWidth,
               const int imageStride, const int imageWidth, const int imageHeight, 
               const float* inputImage, float* outputImage) { 
              
    cout.setf(ios::fixed);
    cout.precision(3); 
              
     //#if aligned
    cout << left << setw(40) << testName;
    for ( list<int>::iterator it = kernels.begin(); it != kernels.end(); it++ ) {
        int kernelWidth = *it; 
        if(kernelWidth >= minKernelWidth && (kernelWidth <= maxKernelWidth || maxKernelWidth == 0)) {
            int kernelStride = calculateAlignedStride(kernelWidth, sizeof(float), ALIGMENT_BYTES);
            float* kernelX = gaussianKernel1D(kernelWidth, 2); 
            float* kernelY = gaussianKernel1D(kernelWidth, 2); 
            clear2DBuffer(outputImage, imageStride, imageHeight);
            float m = 0;
            vector<float> iter;
            for (int i = 0; i < iterations; i++) {
                m_Timer.start();    
                if(testName == "separableConvolve"){
                     separableConvolve (imageStride, imageWidth, imageHeight, 
                                   kernelWidth, inputImage, outputImage, kernelX, kernelY);
                }  
                else if(testName == "sc3SSE") {
                    sc3SSE (imageStride, imageWidth, imageHeight, 
                        inputImage, outputImage, kernelX, kernelY);
                }
                else if(testName == "sc5SSE") {
                    sc5SSE (imageStride, imageWidth, imageHeight, 
                        inputImage, outputImage, kernelX, kernelY);
                }
                else if(testName == "sc7SSE") {
                    sc7SSE (imageStride, imageWidth, imageHeight, 
                        inputImage, outputImage, kernelX, kernelY);
                }
                else if(testName == "scGaussian5SSE") {
                    scGaussian5SSE (imageStride, imageWidth, imageHeight, 
                                    inputImage, outputImage, kernelX);
                }
                else if(testName == "scGaussian7SSE") {
                    scGaussian7SSE (imageStride, imageWidth, imageHeight, 
                                    inputImage, outputImage, kernelX);
                }
                else if(testName == "scGaussian9SSE") {
                    scGaussian9SSE (imageStride, imageWidth, imageHeight, 
                                    inputImage, outputImage, kernelX);
                }
                else if(testName == "sc9SSE") {
                    sc9SSE (imageStride, imageWidth, imageHeight, 
                        inputImage, outputImage, kernelX, kernelY);
                }
                else if(testName == "scSSE") {
                    scSSE (imageStride, imageWidth, imageHeight, kernelWidth,
                        inputImage, outputImage, kernelX, kernelY);
                }
                m_Timer.stop();
                iter.push_back(m_Timer.elapsed()); 
                m += m_Timer.elapsed();       
            }  
            delete[] kernelX;
            delete[] kernelY;
//        assert(assertconvolution(naiveoutputimage, outputimage, imagewidth, imageheight, imagewidth, imagestride, kernelwidth));
            m  = m / iterations;
            cout << setw(7) << m << setw(10) << StDev(iter) << flush; 
        }
        else {
            cout << setw(7) << "-" << setw(10) << "-" << flush; 
        }
        #ifdef DEBUG
        printImage(imageWidth, imageHeight, imageStride, outputImage);
        #endif
        
    }
    cout << endl;

}

void assertTest() {
    cout << "Testing all convolution algorithms..." << endl;
 
    list<string> assertFailList;
    
 
    for(int im = 75; im < 95; im++) {
        for(int k = 3; k < 25; k += 2) {
         
            int imageWidth = im;
            int imageHeight = im;
            
            float* inputImage = allocateFloatAlignedBuffer(imageWidth, imageHeight);
            float* outputImage = allocateFloatAlignedBuffer(imageWidth, imageHeight);
            int imageStride  = calculateAlignedStride(imageWidth, sizeof(float), ALIGMENT_BYTES);
            
            float* naiveInputImage = new float[imageWidth * imageHeight];
            float* naiveOutputImage = new float[imageWidth * imageHeight];
    
            int kernelWidth = k;
            int kernelStride = calculateAlignedStride(kernelWidth, sizeof(float), ALIGMENT_BYTES);
            float* kernel = gaussianKernel2D(kernelWidth, 2);
            float* kernelX = gaussianKernel1D(kernelWidth, 2); 
            float* kernelY = gaussianKernel1D(kernelWidth, 2); 
            
            for (int i = 0; i < imageStride * imageHeight; i++) {
                 inputImage[i] = 0;
            }       
             
            for (int i = 0; i < imageHeight; i++) {
                for (int j = 0; j < imageWidth; j++) {
                    inputImage[i * imageStride + j] = rand() % 5;
                }
            }
         
            for (int i = 0; i < imageWidth * imageHeight; i++) {
                 naiveInputImage[i] = 0;
            }       
            
            PRINT_LINE(); 
            for (int i = 0; i < imageHeight; i++) {
                 for (int j = 0; j < imageWidth; j++) {
                     naiveInputImage[i * imageWidth + j] = inputImage[i * imageStride + j];
                     PRINT_INLINE(naiveInputImage[i * imageWidth + j]); 
                 }
            PRINT_LINE(); 
            }
         
            float *naiveKernel = new float[kernelWidth * kernelWidth];
         
            for (int i = 0; i < kernelWidth * kernelWidth; i++) {
                 naiveKernel[i] = 0;
            }       
         
                 PRINT_LINE(); 
            for (int i = 0; i < kernelWidth; i++) {
                 for (int j = 0; j < kernelWidth; j++) {
                     naiveKernel[i * kernelWidth + j] = kernel[i * kernelStride + j];
                     PRINT_INLINE(naiveKernel[i * kernelWidth + j]); 
                 }
                 PRINT_LINE(); 
             }    
         
             naiveConvolve(imageWidth, imageWidth, imageHeight, 
                             kernelWidth, kernelWidth, naiveInputImage, 
                             naiveOutputImage, naiveKernel);
                             
             //#ifdef DEBUG
                //printImage(kernelWidth, kernelWidth, kernelWidth, kernelWidth, naiveKernel);
//             cout << endl;
//             printImage(imageWidth, imageHeight, imageWidth, kernelWidth, naiveOutputImage);
             //#endif
         
            //#endif
          
            stringstream s;
            s << "Image: " << im << "x" << im << endl;
            s << "Kernel: " << k << endl;
            //ok  
            clear2DBuffer(outputImage, imageStride, imageHeight);
            alignedConvolve(imageStride, imageWidth, imageHeight, 
                            kernelStride, kernelWidth, inputImage, 
                            outputImage, kernel); 
            if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
                stringstream f;
                f << "alignedConvolve fail!" << endl;
                f << s.str();
                assertFailList.push_back(f.str());
            }
 
            //ok  
            clear2DBuffer(outputImage, imageStride, imageHeight);
            unalignedSSEConvolve(imageStride, imageWidth, imageHeight, 
                            kernelStride, kernelWidth, inputImage, 
                            outputImage, kernel); 
            if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
                stringstream f;
                f << "unalignedSSEConvolve fail!" << endl;
                f << s.str();
                assertFailList.push_back(f.str());
            }
 
            //ok  
            clear2DBuffer(outputImage, imageStride, imageHeight);
            sseNoReuse4Convolve(imageStride, imageWidth, imageHeight, 
                            kernelStride, kernelWidth, inputImage, 
                            outputImage, kernel); 
            if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
                stringstream f;
                f << "sseNoReuse4Convolve fail!" << endl;
                f << s.str();
                assertFailList.push_back(f.str());
            }
 
//            clear2DBuffer(outputImage, imageStride, imageHeight);
//            sseNoReuse4Convolve2(imageStride, imageWidth, imageHeight, 
//                            kernelStride, kernelWidth, inputImage, 
//                            outputImage, kernel); 
//            if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
//                stringstream f;
//                f << "sseNoReuse4Convolve2 fail!" << endl;
//                f << s.str();
//                assertFailList.push_back(f.str());
//            }
 
            clear2DBuffer(outputImage, imageStride, imageHeight);
            sseReuse4Convolve(imageStride, imageWidth, imageHeight, 
                            kernelStride, kernelWidth, inputImage, 
                            outputImage, kernel); 
            if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
                stringstream f;
                f << "sseReuse4Convolve fail!" << endl;
                f << s.str();
                assertFailList.push_back(f.str());
            }
            
            clear2DBuffer(outputImage, imageStride, imageHeight);
            sseNoReuse1Convolve(imageStride, imageWidth, imageHeight, 
                            kernelStride, kernelWidth, inputImage, 
                            outputImage, kernel); 
            if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
                stringstream f;
                f << "sseNoReuse1Convolve fail!" << endl;
                f << s.str();
                assertFailList.push_back(f.str());
            }
 
            clear2DBuffer(outputImage, imageStride, imageHeight);
            sseReuse1Convolve(imageStride, imageWidth, imageHeight, 
                            kernelStride, kernelWidth, inputImage, 
                            outputImage, kernel); 
            if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
                stringstream f;
                f << "sseReuse1Convolve fail!" << endl;
                f << s.str();
                assertFailList.push_back(f.str());
            }
 
             
            clear2DBuffer(outputImage, imageStride, imageHeight);
            sseNoReuse2Convolve(imageStride, imageWidth, imageHeight, 
                            kernelStride, kernelWidth, inputImage, 
                            outputImage, kernel); 
            if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
                stringstream f;
                f << "sseNoReuse2Convolve fail!" << endl;
                f << s.str();
                assertFailList.push_back(f.str());
            }
 
            clear2DBuffer(outputImage, imageStride, imageHeight);
            sseReuse2Convolve(imageStride, imageWidth, imageHeight, 
                            kernelStride, kernelWidth, inputImage, 
                            outputImage, kernel); 
            if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
                stringstream f;
                f << "sseReuse2Convolve fail!" << endl;
                f << s.str();
                assertFailList.push_back(f.str());
            }
            

            clear2DBuffer(outputImage, imageStride, imageHeight);
            sseNoReuse3Convolve(imageStride, imageWidth, imageHeight, 
                            kernelStride, kernelWidth, inputImage, 
                            outputImage, kernel); 
            if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
                stringstream f;
                f << "sseNoReuse3Convolve fail!" << endl;
                f << s.str();
                assertFailList.push_back(f.str());
            }
 
            clear2DBuffer(outputImage, imageStride, imageHeight);
            sseReuse3Convolve(imageStride, imageWidth, imageHeight, 
                            kernelStride, kernelWidth, inputImage, 
                            outputImage, kernel); 
            if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
                stringstream f;
                f << "sseReuse3Convolve fail!" << endl;
                f << s.str();
                assertFailList.push_back(f.str());
            }
            

            clear2DBuffer(outputImage, imageStride, imageHeight);
            sseNoReuse5Convolve(imageStride, imageWidth, imageHeight, 
                            kernelStride, kernelWidth, inputImage, 
                            outputImage, kernel); 
            if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
                stringstream f;
                f << "sseNoReuse5Convolve fail!" << endl;
                f << s.str();
                assertFailList.push_back(f.str());
            }
 
            clear2DBuffer(outputImage, imageStride, imageHeight);
            sseReuse5Convolve(imageStride, imageWidth, imageHeight, 
                            kernelStride, kernelWidth, inputImage, 
                            outputImage, kernel); 
            if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
                stringstream f;
                f << "sseReuse5Convolve fail!" << endl;
                f << s.str();
                assertFailList.push_back(f.str());
            }
            
            clear2DBuffer(outputImage, imageStride, imageHeight);
            sseNoReuse6Convolve(imageStride, imageWidth, imageHeight, 
                            kernelStride, kernelWidth, inputImage, 
                            outputImage, kernel); 
            if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
                stringstream f;
                f << "sseNoReuse6Convolve fail!" << endl;
                f << s.str();
                assertFailList.push_back(f.str());
            }
 
            clear2DBuffer(outputImage, imageStride, imageHeight);
            sseReuse6Convolve(imageStride, imageWidth, imageHeight, 
                            kernelStride, kernelWidth, inputImage, 
                            outputImage, kernel); 
            if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
                stringstream f;
                f << "sseReuse6Convolve fail!" << endl;
                f << s.str();
                assertFailList.push_back(f.str());
            }
            
  
              clear2DBuffer(outputImage, imageStride, imageHeight);
            sseNoReuse7Convolve(imageStride, imageWidth, imageHeight, 
                            kernelStride, kernelWidth, inputImage, 
                            outputImage, kernel); 
            if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
                stringstream f;
                f << "sseNoReuse7Convolve fail!" << endl;
                f << s.str();
                assertFailList.push_back(f.str());
            }
 
            clear2DBuffer(outputImage, imageStride, imageHeight);
            sseReuse7Convolve(imageStride, imageWidth, imageHeight, 
                            kernelStride, kernelWidth, inputImage, 
                            outputImage, kernel); 
            if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
                stringstream f;
                f << "sseReuse7Convolve fail!" << endl;
                f << s.str();
                assertFailList.push_back(f.str());
            }
            

            clear2DBuffer(outputImage, imageStride, imageHeight);
            pointerArithmeticConvolve(imageStride, imageWidth, imageHeight, 
                            kernelStride, kernelWidth, inputImage, 
                            outputImage, kernel); 
            if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
                stringstream f;
                f << "pointerArithmeticConvolve fail!" << endl;
                f << s.str();
                assertFailList.push_back(f.str());
            }
      
            if(imageWidth > 128 &&  imageHeight > 128) {                        
                clear2DBuffer(outputImage, imageStride, imageHeight);
                loopUnrollConvolve(imageStride, imageWidth, imageHeight, 
                                kernelStride, kernelWidth, inputImage, 
                                outputImage, kernel); 
                if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
                    stringstream f;
                    f << "loopUnrollConvolve fail!" << endl;
                    f << s.str();
                    assertFailList.push_back(f.str());
                }
            }
    
            if(imageWidth > 128 &&  imageHeight > 128) {                        
                clear2DBuffer(outputImage, imageStride, imageHeight);
                prefetchConvolve64(imageStride, imageWidth, imageHeight, 
                                kernelStride, kernelWidth, inputImage, 
                                outputImage, kernel); 
                if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
                    stringstream f;
                    f << "prefetchConvolve64 fail!" << endl;
                    f << s.str();
                    assertFailList.push_back(f.str());
                }
            }
            
            if(imageWidth > 128 &&  imageHeight > 128) {                        
                clear2DBuffer(outputImage, imageStride, imageHeight);
                prefetchConvolve128(imageStride, imageWidth, imageHeight, 
                                kernelStride, kernelWidth, inputImage, 
                                outputImage, kernel); 
                if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
                    stringstream f;
                    f << "prefetchConvolve128 fail!" << endl;
                    f << s.str();
                    assertFailList.push_back(f.str());
                }
            }    
            
            clear2DBuffer(outputImage, imageStride, imageHeight);
            opConvolve(imageStride, imageWidth, imageHeight, 
                            kernelStride, kernelWidth, inputImage, 
                            outputImage, kernel); 
            if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
                stringstream f;
                f << "opConvolve fail!" << endl;
                f << s.str();
                assertFailList.push_back(f.str());
            }
     
            if(kernelWidth > 11) {                        
                clear2DBuffer(outputImage, imageStride, imageHeight);
                sseWideKernelConvolve(imageStride, imageWidth, imageHeight, 
                                kernelStride, kernelWidth, inputImage, 
                                outputImage, kernel); 
                if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
                    stringstream f;
                    f << "sseWideKernelConvolve fail!" << endl;
                    f << s.str();
                    assertFailList.push_back(f.str());
                }
            }
            
            if(kernelWidth == 3) {                        
                clear2DBuffer(outputImage, imageStride, imageHeight);
                sse3Convolve(imageStride, imageWidth, imageHeight, 
                            kernelStride, inputImage, 
                            outputImage, kernel);
                if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
                    stringstream f;
                    f << "sse3Convolve fail!" << endl;
                    f << s.str();
                    //printImage(imageWidth, imageHeight, imageWidth, kernelWidth, naiveOutputImage);
                    assertFailList.push_back(f.str());
                }
            }
                        
            if(kernelWidth == 3) {                        
                clear2DBuffer(outputImage, imageStride, imageHeight);
                sse3CmConvolve(imageStride, imageWidth, imageHeight, 
                            kernelStride, inputImage, 
                            outputImage, kernel);                 
                if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
                    stringstream f;
                    f << "sse3CmConvolve fail!" << endl;
                    f << s.str();
                    //printImage(imageWidth, imageHeight, imageWidth, kernelWidth, naiveOutputImage);
                    assertFailList.push_back(f.str());
                }
            }     
            
            if(kernelWidth == 3) {                        
                clear2DBuffer(outputImage, imageStride, imageHeight);
                //cout << "clear2DBuffer" << endl;
                //printImage(imageWidth, imageHeight, imageStride, outputImage);
                //cout << "end clear2DBuffer" << endl;
                sse3LbConvolve(imageStride, imageWidth, imageHeight, 
                            kernelStride, inputImage, 
                            outputImage, kernel);
                if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
                    stringstream f;
                    f << "sse3LbConvolve fail!" << endl;
                    f << s.str();
                    //cout << "naiveOutputImage" << endl;
                    //printImage(imageWidth, imageHeight, imageWidth, kernelWidth, naiveOutputImage);
                    //cout << "end naiveOutputImage" << endl;
                    assertFailList.push_back(f.str());
                }
            }
                        
            if(kernelWidth == 5) {                        
                clear2DBuffer(outputImage, imageStride, imageHeight);
                sse5Convolve(imageStride, imageWidth, imageHeight, 
                            kernelStride, inputImage, 
                            outputImage, kernel);
                if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
                    stringstream f;
                    f << "sse5Convolve fail!" << endl;
                    f << s.str();
//                    printImage(imageWidth, imageHeight, imageWidth, kernelWidth, naiveOutputImage);
                    assertFailList.push_back(f.str());
                }
            }
                        
            if(kernelWidth == 7) {
                clear2DBuffer(outputImage, imageStride, imageHeight);
                sse7Convolve(imageStride, imageWidth, imageHeight, 
                            kernelStride, inputImage, 
                            outputImage, kernel);
                if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
                    stringstream f;
                    f << "sse7Convolve fail!" << endl;
                    f << s.str();
//                    printImage(imageWidth, imageHeight, imageWidth, kernelWidth, naiveOutputImage);
                    assertFailList.push_back(f.str());
                }
            }
                        
            if(kernelWidth == 9) {
                clear2DBuffer(outputImage, imageStride, imageHeight);
                sse9Convolve(imageStride, imageWidth, imageHeight, 
                            kernelStride, inputImage, 
                            outputImage, kernel);
                if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
                    stringstream f;
                    f << "sse9Convolve fail!" << endl;
                    f << s.str();
//                    printImage(imageWidth, imageHeight, imageWidth, kernelWidth, naiveOutputImage);
                    assertFailList.push_back(f.str());
                }
            }
                        
            if(kernelWidth == 11) {                        
                clear2DBuffer(outputImage, imageStride, imageHeight);
                sse11Convolve(imageStride, imageWidth, imageHeight, 
                            kernelStride, inputImage, 
                            outputImage, kernel);
                if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
                    stringstream f;
                    f << "sse11Convolve fail!" << endl;
                    f << s.str();
//                    printImage(imageWidth, imageHeight, imageWidth, kernelWidth, naiveOutputImage);
                    assertFailList.push_back(f.str());
                }
            }           
            
            clear2DBuffer(outputImage, imageStride, imageHeight);
            separableConvolve (imageStride, imageWidth, imageHeight, 
                        kernelWidth, inputImage, outputImage, kernelX, kernelY);
            if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
                stringstream f;
                f << "separableConvolve fail!" << endl;
                f << s.str();
                assertFailList.push_back(f.str());
            }
                        
            clear2DBuffer(outputImage, imageStride, imageHeight);
            scSSE (imageStride, imageWidth, imageHeight, 
                   kernelWidth, inputImage, outputImage, kernelX, kernelY);
            if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
                stringstream f;
                f << "scSSE fail!" << endl;
                f << s.str();
                assertFailList.push_back(f.str());
            }
            
            if(kernelWidth == 3) {
                clear2DBuffer(outputImage, imageStride, imageHeight);
                sc3SSE (imageStride, imageWidth, imageHeight, 
                            inputImage, outputImage, kernelX, kernelY);
                if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
                    stringstream f;
                    f << "sc3SSE fail!" << endl;
                    f << s.str();
                    assertFailList.push_back(f.str());
                }
            }           
            
            if(kernelWidth == 5) {
                clear2DBuffer(outputImage, imageStride, imageHeight);
                sc5SSE (imageStride, imageWidth, imageHeight, 
                            inputImage, outputImage, kernelX, kernelY);
                if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
                    stringstream f;
                    f << "sc5SSE fail!" << endl;
                    f << s.str();
                    assertFailList.push_back(f.str());
                }
            }
            
            if(kernelWidth == 7) {
                clear2DBuffer(outputImage, imageStride, imageHeight);
                sc7SSE (imageStride, imageWidth, imageHeight, 
                            inputImage, outputImage, kernelX, kernelY);
                if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
                    stringstream f;
                    f << "sc7SSE fail!" << endl;
                    f << s.str();
                    assertFailList.push_back(f.str());
                }
            }
            
            if(kernelWidth == 9) {                        
                clear2DBuffer(outputImage, imageStride, imageHeight);
                sc9SSE (imageStride, imageWidth, imageHeight, 
                            inputImage, outputImage, kernelX, kernelY);
                if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
                    stringstream f;
                    f << "sc9SSE fail!" << endl;
                    f << s.str();
                    //printImage(imageWidth, imageHeight, imageWidth, naiveOutputImage);
                    assertFailList.push_back(f.str());
                }
            }    
            

//            for(int y = 16; y <= 512; y += 16) {
//                for(int x = 16; x <= 512; x += 16) {
//                    clear2DBuffer(outputImage, imageStride, imageHeight);
//                    loopBlockAlignedSSEConvolve(imageStride, imageWidth, imageHeight, 
//                                 kernelStride, kernelWidth, inputImage, 
//                                 outputImage, kernel, x, y);
//                    if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
//                        stringstream f;
//                        f << "loopBlockAlignedSSEConvolve fail!" << endl;
//                        f << s.str();
//                        //printImage(imageWidth, imageHeight, imageWidth,  naiveOutputImage);
//                        assertFailList.push_back(f.str());
//                    }
//                }
//            }
            
            delete[] naiveInputImage;
            delete[] inputImage;
            delete[] outputImage;
        }
    }
    
    if(assertFailList.size() > 0) {
        cout << "Some test(s) failed." << endl;
        for ( list<string>::iterator it = assertFailList.begin(); it != assertFailList.end(); it++ ) {
            cout << *it << endl;
        }
    }
    else {
        cout << "All tests has passed!" << endl;
    }
    
    cout << endl;
    
}

struct ImageSize {
    int width;
    int height;
    ImageSize(int w, int h) {
        width = w;
        height = h;
    }
};

struct LoopBlockConfig {
    vector<ImageSize> images;
    vector<int> kernels;
    LoopBlockConfig(vector<ImageSize>& imgs, vector<int>& krnls) {
        images = imgs;
        kernels = krnls;
    }
};

void loopBlockAlignedSSEConvolveTest() {
 
    string line;
    ifstream confFile ("lbTest.txt");
    vector<LoopBlockConfig> lbcs;
    
    
    if (confFile.is_open()) {
        vector<int> kernels;
        vector<ImageSize> images;
        while ( confFile.good() )
        {
            //images
            getline (confFile, line);
            cout << "reading line: " << line << endl << flush;
            vector<string> imgs = split(line, ',');
            
            for(int i = 0; i < imgs.size(); ++i) {
                vector<string> tmp = split(imgs[i], 'x'); 
                images.push_back(ImageSize(atoi(tmp[0].c_str()), atoi(tmp[1].c_str())));
            }
            //kernels
            getline (confFile, line);
            cout << "reading line: " << line << endl << flush;
            vector<string> krnls = split(line, ',');
            for(int i = 0; i < krnls.size(); ++i) {
                kernels.push_back(atoi(krnls[i].c_str()));
            }
        }
        confFile.close();
        lbcs.push_back(LoopBlockConfig(images, kernels)); 
    }
    else {
        cout << "Unable to open file" << flush << endl; 
        return;
    }
    
    cout << "Wait until tests are finished..." << endl << flush;
    
    const string& file = "lbTest.csv";
    ofstream outFile;
    outFile.open (file.c_str(), std::ios::out);
    
    for(int i = 0; i < lbcs.size(); i++) {
        LoopBlockConfig lbc = lbcs[i];
        for(int img = 0; img < lbc.images.size(); ++img) {
            ImageSize is = lbc.images[img];
            const int imageWidth = is.width; 
            const int imageHeight = is.height; 
            int imageStride  = calculateAlignedStride(imageWidth, sizeof(float), ALIGMENT_BYTES);            
            float* inputImage = allocateFloatAlignedBuffer(imageWidth, imageHeight);
            populateBuffer(imageStride, imageWidth, imageHeight, inputImage); 
            float* outputImage = allocateFloatAlignedBuffer(imageWidth, imageHeight);
            for (int krnl = 0; krnl < lbc.kernels.size(); ++krnl) {
                const int kernelWidth = lbc.kernels[krnl];
                outFile << "image: " <<  imageWidth << "x" << imageHeight << endl; 
                outFile << "kernel: " <<  kernelWidth << "x" << kernelWidth << endl; 
                const int kernelStride = calculateAlignedStride(kernelWidth, sizeof(float), ALIGMENT_BYTES);
                float* kernel = gaussianKernel2D(kernelWidth, 2);


                outFile << "xy,";
                for(int x = 16; x <= 512; x += 16) {
                    //x == 128 ? s << x : s << x << ",";
                    x == 512 ? outFile << x : outFile << x << ",";
                }
                outFile << endl;
                for(int y = 16; y <= 512; y += 16) {
                    outFile <<  y << ","; 
                    for(int x = 16; x <= 512; x += 16) {
                        float m = 0;
                        for (int i = 0; i < 10; i++) {
                            m_Timer.start();    
                            loopBlockAlignedSSEConvolve(imageStride, imageWidth, imageHeight, 
                                         kernelStride, kernelWidth, inputImage, 
                                         outputImage, kernel, x, y);
                            m_Timer.stop();
                            m += m_Timer.elapsed();     
                        }
//                        if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth))
//                        {
//                            cout << "zica"; 
//                        }        
                        float time = m / 10;
                        x == 512 ? outFile << time : outFile << time << ",";
                    }
                    outFile << endl;
                }
                delete[] kernel;
            }
            delete[] inputImage;
            delete[] outputImage;
        }
    }
    outFile.close();
    cout << "Tests are finished. See lbTest.csv" << endl << flush;
} 


void naiveConvolveTest( const int iterations, list<int>& kernels, 
                        const int imageWidth, const int imageHeight, 
                        const int kernelWidth, 
                        const float* inputImage, const float* kernel) {

    //#if NAIVE
    cout << left << setw(40) << "naiveConvolve";
    float* naiveInputImage = new float[imageWidth * imageHeight];
    float* naiveOutputImage = new float[imageWidth * imageHeight];
    for ( list<int>::iterator it = kernels.begin(); it != kernels.end(); it++ ) {
        const int kernelWidth = *it;
        kernel = gaussianKernel2D(kernelWidth, 2);
        const int kernelStride = calculateAlignedStride(kernelWidth, sizeof(float), ALIGMENT_BYTES);
        for (int i = 0; i < imageWidth * imageHeight; i++) {
             naiveInputImage[i] = 0;
        }       
        
        int imageStride  = calculateAlignedStride(imageWidth, sizeof(float), ALIGMENT_BYTES);
        
        PRINT_LINE(); 
        for (int i = 0; i < imageHeight; i++) {
             for (int j = 0; j < imageWidth; j++) {
                 naiveInputImage[i * imageWidth + j] = inputImage[i * imageStride + j];
                 PRINT_INLINE(naiveInputImage[i * imageWidth + j]); 
             }
        PRINT_LINE(); 
        }
     
        float *naiveKernel = new float[kernelWidth * kernelWidth];
     
        for (int i = 0; i < kernelWidth * kernelWidth; i++) {
             naiveKernel[i] = 0;
        }       
     
             PRINT_LINE(); 
        for (int i = 0; i < kernelWidth; i++) {
             for (int j = 0; j < kernelWidth; j++) {
                 naiveKernel[i * kernelWidth + j] = kernel[i * kernelStride + j];
                 PRINT_INLINE(naiveKernel[i * kernelWidth + j]); 
             }
             PRINT_LINE(); 
         }    
     
         float m = 0;
         vector<float> iter;
         for (int i = 0; i < iterations; i++) {
             m_Timer.start();    
             naiveConvolve(imageWidth, imageWidth, imageHeight, 
                             kernelWidth, kernelWidth, naiveInputImage, 
                             naiveOutputImage, naiveKernel);
             m_Timer.stop();
              iter.push_back(m_Timer.elapsed()); 
             m += m_Timer.elapsed();     
         }
            delete[] kernel;
         //cout << setw(10) << "-";        
            m  = m / iterations;
            cout << setw(7) << m << setw(10) << StDev(iter) << flush; 
         //#ifdef DEBUG
            //printImage(kernelWidth, kernelWidth, kernelWidth, kernelWidth, naiveKernel);
         //printImage(imageWidth, imageHeight, imageWidth, naiveOutputImage);
         //#endif
     
        //#endif
    }
    cout << endl;
    //#endif
  
    delete[] naiveOutputImage;
    delete[] naiveInputImage;
 
}



//     #if ALIGNEDSSE2CONVOLVE
//     loopBlockConvolve(imageStride, imageWidth, imageHeight, 
//                 kernelStride, kernelWidth, inputImage, 
//                 outputImage, kernel, x, y);
//    #endif 

int main (int argc, char *argv[])
{
    #ifdef __SSE4_1__
        cout << "Running in AMD architecture..." << endl;
    #endif
    
//    PRINT_VECTOR_TRACE(amm_dp_ps(_mm_set1_ps(0.1), _mm_set1_ps(0.2), 254));
//    PRINT_VECTOR_TRACE(_mm_dp_ps(_mm_set1_ps(0.1), _mm_set1_ps(0.2), 254));
// 
//    PRINT_VECTOR_TRACE(amm_blend_ps(_mm_set1_ps(0.1), _mm_set1_ps(0.2), 1));
//    PRINT_VECTOR_TRACE(_mm_blend_ps(_mm_set1_ps(0.1), _mm_set1_ps(0.2), 1)); 
 
    /* initialize random seed: */
    srand ( time(NULL) );
    cout.setf(ios::fixed);
    cout.precision(3); 
    
    int optind=1;
    // decode arguments
    while ((optind < argc) && (argv[optind][0]=='-')) {
        string sw = argv[optind];
        if (sw=="-a") {
            assertTest(); 
            return 0;
        }
        else if (sw=="-lb") {
            loopBlockAlignedSSEConvolveTest(); 
            return 0;
        }
        optind++;
    }    
    list<int> kernels;
    
    cout << "image size " << atoi(argv[1]) << "x" << atoi(argv[2]) << endl;
    cout << left << setw(40) << "algorithm/kernel size";
    for (int i = 4; i < argc; ++i) {
        cout << setw(7) << atoi(argv[i]) << setw(10) << "dv"; 
        kernels.push_back(atoi(argv[i]));
    }
    cout << endl;
    
   
    int imageWidth = atoi(argv[1]); PRINT(imageWidth);
    int imageHeight = atoi(argv[2]); PRINT(imageHeight);
    int iterations = atoi(argv[3]);
    
    float* inputImage = allocateFloatAlignedBuffer(imageWidth, imageHeight);
    float* outputImage = allocateFloatAlignedBuffer(imageWidth, imageHeight);
    int imageStride  = calculateAlignedStride(imageWidth, sizeof(float), ALIGMENT_BYTES);
    
    float k = 1;

    for (int i = 0; i < imageStride * imageHeight; i++) {
         inputImage[i] = 0;
    }       
     
    for (int i = 0; i < imageHeight; i++) {
        for (int j = 0; j < imageWidth; j++) {
            inputImage[i * imageStride + j] = k++;//rand() % 5;
        }
    }
    
    //printImage(imageWidth, imageHeight, imageStride, kernelWidth - 2, inputImage);
    
 
    //float* kernel = allocateFloatAlignedBuffer64(kernelWidth, kernelWidth);
 
//    for (int i = 0; i < kernelStride * kernelWidth; i++) {
//         kernel[i] = 0;
//    }       

    k = 1;

//    for (int i = 0; i < kernelWidth; i++) {
//        for (int j = 0; j < kernelWidth; j++) {
//            kernel[i * kernelStride + j] = rand() % 5;
//        }
//    }    
 
//    kernel[0] = 1;                      kernel[1] = 0;                      kernel[2] = -1;
//    kernel[kernelStride] = 2;           kernel[kernelStride + 1] = 0;       kernel[kernelStride + 2] = -2;
//    kernel[kernelStride * 2] = 1;       kernel[kernelStride * 2 + 1] = 0;   kernel[kernelStride * 2 + 2] = -1;
 
    int kernelWidth = 0;
    int kernelStride = 0;
    float* kernel;
     
//     cout << sysconf (_SC_LEVEL1_DCACHE_LINESIZE) << endl; //retorna o tamanho da linha da memória cache
//     cout << _MM_SHUFFLE(1,0,3,2);
     //gcc -DCLS=$(getconf LEVEL1_DCACHE_LINESIZE) ...

    //#if NAIVECONVOLVE
    naiveConvolveTest (iterations, kernels, imageWidth, imageHeight, 
                       kernelWidth, inputImage, kernel);
    //#endif
    
  
    //#if ALIGNEDCONVOLVE
    run2DTest (alignedConvolve, "alignedConvolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
               inputImage, outputImage);
    //#endif
 
    //#if UNALIGNEDSSECONVOLVE
    run2DTest (unalignedSSEConvolve, "unalignedSSEConvolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage) ;
    //#endif
    

    //#if POINTERARITHMETICCONVOLVE
    run2DTest (pointerArithmeticConvolve, "pointerArithmeticConvolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage);
    //#endif
      
               
    //#if LOOPUNROLLCONVOLVE
    run2DTest (loopUnrollConvolve, "loopUnrollConvolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage);
    //#endif
    
    //#if PREFETCHCONVOLVE64
    run2DTest (prefetchConvolve64, "prefetchConvolve64", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage);
    //#endif
    
    //#if PREFETCHCONVOLVE128
    run2DTest (prefetchConvolve128, "prefetchConvolve128", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage);
    //#endif    

    //#if LOOPBLOCKALIGNEDSSECONVOLVE
    runLoopBlockConvolveTest("loopBlockLoopUnroll16x64Convolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage, 16, 64);
    //#endif

    //#if ALIGNEDSSE1CONVOLVE
    run2DTest (sseNoReuse1Convolve, "sseNoReuse1Convolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage);
    //#endif

    //#if ALIGNEDSSE2CONVOLVE
    run2DTest (sseNoReuse2Convolve, "sseNoReuse2Convolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage);
    //#endif

    //#if ALIGNEDSSE3CONVOLVE
    run2DTest (sseNoReuse3Convolve, "sseNoReuse3Convolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage);
    //#endif

    //#if ALIGNEDSSENOREUSE4SUMSCONVOLVE
    run2DTest (sseNoReuse4Convolve, "sseNoReuse4Convolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage);
    //#endif
    
    //#if ALIGNEDSSE5CONVOLVE
    run2DTest (sseNoReuse5Convolve, "sseNoReuse5Convolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage);
    //#endif
    
     //#if ALIGNEDSSE6CONVOLVE
    run2DTest (sseNoReuse6Convolve, "sseNoReuse6Convolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage);
    //#endif
    
     //#if ALIGNEDSSE7CONVOLVE
    run2DTest (sseNoReuse7Convolve, "sseNoReuse7Convolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage);
    //#endif
    

    //#if ALIGNEDSSEREUSE3SUMSCONVOLVE
    run2DTest (sseReuse1Convolve, "sseReuse1Convolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage);
    //#endif
    
    //#if ALIGNEDSSEREUSE3SUMSCONVOLVE
    run2DTest (sseReuse2Convolve, "sseReuse2Convolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage);
    //#endif


    //#if ALIGNEDSSEREUSE3SUMSCONVOLVE
    run2DTest (sseReuse3Convolve, "sseReuse3Convolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage);
    //#endif
 
    //#if ALIGNEDSSEREUSE4SUMSCONVOLVE
    run2DTest (sseReuse4Convolve, "sseReuse4Convolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage);
    //#endif

    //#if ALIGNEDSSEREUSE4SUMSCONVOLVE
    run2DTest (sseReuse5Convolve, "sseReuse5Convolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage);
    //#endif

    //#if ALIGNEDSSEREUSE4SUMSCONVOLVE
    run2DTest (sseReuse6Convolve, "sseReuse6Convolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage);
    //#endif

    //#if ALIGNEDSSEREUSE4SUMSCONVOLVE
    run2DTest (sseReuse7Convolve, "sseReuse7Convolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage);
    //#endif


    //#if LOOPBLOCKALIGNEDSSECONVOLVE
    runLoopBlockConvolveTest("loopBlock128x128Convolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage, 128, 128);
    //#endif




    //#if LOOPBLOCKALIGNEDSSECONVOLVE
    runLoopBlockConvolveTest("loopBlock16x48AlignedSSEConvolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage, 16, 48);
    //#endif

    //#if LOOPBLOCKALIGNEDSSECONVOLVE
    runLoopBlockConvolveTest("loopBlock32x48AlignedSSEConvolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage, 32, 48);
    //#endif

    //#if LOOPBLOCKALIGNEDSSECONVOLVE
    runLoopBlockConvolveTest("loopBlock32x32AlignedSSEConvolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage, 32, 32);
    //#endif


    //#if LOOPBLOCKALIGNEDSSECONVOLVE
    runLoopBlockConvolveTest("loopBlock64x16AlignedSSEConvolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage, 64, 16);
    //#endif


      
    //#if LOOPBLOCKALIGNEDSSECONVOLVE
    runLoopBlockConvolveTest("loopBlock64x48AlignedSSEConvolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage, 64, 48);
    //#endif

    //#if LOOPBLOCKALIGNEDSSECONVOLVE
    runLoopBlockConvolveTest("loopBlock64x64AlignedSSEConvolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage, 64, 64);
    //#endif

    //#if LOOPBLOCKALIGNEDSSECONVOLVE
    runLoopBlockConvolveTest("loopBlock80x48AlignedSSEConvolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage, 80, 48);
    //#endif

    //#if LOOPBLOCKALIGNEDSSECONVOLVE
    runLoopBlockConvolveTest("loopBlock128x48AlignedSSEConvolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage, 128, 48);
    //#endif

    //#if LOOPBLOCKALIGNEDSSECONVOLVE
    runLoopBlockConvolveTest("loopBlock128x128AlignedSSEConvolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage, 128, 128);
    //#endif

    //#if LOOPBLOCKALIGNEDSSECONVOLVE
    runLoopBlockConvolveTest("loopBlock512x48AlignedSSEConvolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage, 512, 48);
    //#endif

 
    //#if LOOPBLOCKALIGNEDSSECONVOLVE
    runLoopBlockConvolveTest("loopBlock512x256AlignedSSEConvolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage, 512, 256);
    //#endif

 
     
    //#if OPCONVOLVE
    run2DTest (opConvolve, "opConvolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage);
    //#endif
     
    //#if SSEWIDEKERNELCONVOLVE
    run2DTest (sseWideKernelConvolve, "sseWideKernelConvolve", iterations, kernels, 13, 0, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage);
    //#endif
     
//    #if SEPARABLECONVOLVE
    runScTest ("separableConvolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage);
//    #endif 

     
    //#if SC3SSE
    runScTest ("sc3SSE", iterations, kernels, 3, 3, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage);
    //#endif    
     
    //#if SC5SSE
    runScTest ("sc5SSE", iterations, kernels, 5, 5, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage);
    //#endif    
     
    //#if SC7SSE
    runScTest ("sc7SSE", iterations, kernels, 7, 7, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage);
    //#endif    
     
    //#if SC9SSE
    runScTest ("sc9SSE", iterations, kernels, 9, 9, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage);
    //#endif    
     
    //#if SCSSE 
    runScTest ("scSSE", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage);
    //#endif   
     
    //#if SCGAUSSIAN5SSE 
    runScTest ("scGaussian5SSE", iterations, kernels, 5, 5, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage);
    //#endif   
        
    //#if SCGAUSSIAN&SSE 
    runScTest ("scGaussian7SSE", iterations, kernels, 7, 7, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage);
    //#endif   
        
    //#if SCGAUSSIAN9SSE 
    runScTest ("scGaussian9SSE", iterations, kernels, 9, 9, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage);
    //#endif    
    
    //#if SSE3CONVOLVE
    runSSETest ("sse3Convolve", iterations, kernels, 3, 3, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage);
    //#endif    
    
    //#if SSE3CMCONVOLVE
    runSSETest ("sse3CmConvolve", iterations, kernels, 3, 3, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage);
    //#endif    
    
    //#if SSE3LBCONVOLVE
    runSSETest ("sse3LbConvolve", iterations, kernels, 3, 3, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage);
    //#endif    
    
    //#if SSE5CONVOLVE
    runSSETest ("sse5Convolve", iterations, kernels, 5, 5, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage);
    //#endif    
    
    //#if SSE7CONVOLVE
    runSSETest ("sse7Convolve", iterations, kernels, 7, 7, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage);
    //#endif    
    
    //#if SSE9CONVOLVE
    runSSETest ("sse9Convolve", iterations, kernels, 9, 9, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage);
    //#endif    
    
    //#if SSE11CONVOLVE
    runSSETest ("sse11Convolve", iterations, kernels, 11, 11, imageStride, imageWidth, imageHeight, 
             inputImage, outputImage);
    //#endif    
    
    
    delete[] inputImage;
    delete[] outputImage;
    //delete[] kernel;

    return 0;
}



static bool assertConvolution(const float* controlOutput, const float* convolveOutput,
                           int imageWidth, int imageHeight, int controlStride, 
                           int convolveStride, int kernelWidth) {
    int stopX = imageWidth - 2 * (kernelWidth / 2);
    int stopY = imageHeight - 2 * (kernelWidth / 2);
        for (int y = 0; y < stopY; ++y) {
            for (int x = 0; x < stopX; ++x) {
                float a = controlOutput[y * controlStride + x];
                float b = convolveOutput[y * convolveStride + x];
                
                //a = floor(a * 10000);
                //b = floor(b * 10000);
                //b = (int)a >> 3 << 3;
                if ((b == 0 && a != 0) ||  1 - (a / b) > 0.005 ) {
//                    cout << "w " << imageWidth << "  h " << imageHeight << "  kw " << kernelWidth << endl; 
//                    cout << "a,b " << a << " " << b << endl;
//                    cout << "abs " << 1 - (a / b) << endl;
//                    cout << "image" << endl;
//                    printImage(imageWidth, imageHeight, convolveStride, kernelWidth, convolveOutput);
//                    cout << "end image" << endl;
               // cout << a << " " << b << endl;
                //cout << "Assert false\n" << flush;
                    return false;
                } 
            }
        }   
 //cout << "Assert true\n" << flush;
    return true;
}




float* gaussianKernel2D(const int kernelWidth, const float sigma) {
 
    
    const int radius = kernelWidth / 2;
 
    float* kernel = allocateFloatAlignedBuffer(kernelWidth, kernelWidth);
    int kernelStride = calculateAlignedStride(kernelWidth, sizeof(float), ALIGMENT_BYTES);

     for (int i = 0; i < kernelStride * kernelWidth; i++) {
         kernel[i] = 0;
    }       
    
    #ifdef DEBUG 
    cout << endl;
    cout << "Gaussian2D kernel" << endl;
    cout << "kernelWidth " << kernelWidth << endl;
    #endif
    
    //http://homepages.inf.ed.ac.uk/rbf/HIPR2/gsmooth.htm
    for(int y = -radius; y < radius + 1; ++y) {
        for(int x = -radius; x < radius + 1; ++x) {
            float value = exp( (pow(x,2) + pow(y,2)) / (-2 * pow(sigma, 2))) / (2 * M_PI * pow(sigma, 2));       
            kernel[(y + radius) * kernelStride + (x + radius)] = value;
            #ifdef DEBUG 
            cout << value << " ";
            #endif
        }
        #ifdef DEBUG 
        cout << endl;
        #endif
    }
    return kernel;
}


float* gaussianKernel1D(const int kernelWidth, const float sigma) {
 
    
    const int radius = kernelWidth / 2;
 
    float* kernel __attribute__ ((aligned(ALIGMENT_BYTES))) = new float[kernelWidth + 4 - (kernelWidth % 4)]; 
//            cout << kernelWidth + 4 - (kernelWidth % 4)  << " ";

    for (int i = 0; i < kernelWidth + 4 - (kernelWidth % 4); i++) {
     
         kernel[i] = 0;
    }       
    
    //http://homepages.inf.ed.ac.uk/rbf/HIPR2/gsmooth.htm
    for(int x = -radius; x < radius + 1; ++x) {
            float value = exp ( pow ( x, 2 ) / ( -2 * pow ( sigma, 2 ) ) ) / ( sqrt( 2 * M_PI ) * sigma );
            kernel[x + radius] = value;
//            cout << value << " ";
    }
    return kernel;
}

