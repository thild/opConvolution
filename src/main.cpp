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
#include "StopWatch.h"
#include "statistics.h"
#include "util.h"
#include <assert.h>
#include <list>
#include <map>
#include <omp.h>

#include <fstream>
#include <math.h>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/foreach.hpp>

#define foreach BOOST_FOREACH

#include "opConvolutionFilter.h"
#include <valgrind/callgrind.h>
                         
using namespace std;
using boost::lexical_cast;
using boost::bad_lexical_cast;  

StopWatch m_StopWatch; 
map<int, double> naiveTimes;
map<int, double> separableTimes;

void loopBlockAlignedSSEConvolveTest();

static bool assertConvolution(const float* controlOutput, const float* convolveOutput,
                           int imageWidth, int imageHeight, int controlStride, 
                           int convolveStride, int kernelWidth);
                           
static void prepareTestBuffers (const int imageStride, const int imageHeight, 
                                float* inputImage, float* outputImage);

             
static void clearCache();
                                  
float* gaussianKernel2D(const int kernelWidth, const float sigma);
float* gaussianKernel1D(const int kernelWidth, const float sigma);


ostream& tab( ostream& output ) { return output << '\t'; } 


void run2DTest(void (*convolutionFunction)(const int imageStride, const int imageWidth, const int imageHeight, 
                                         const int kernelStride, const int kernelWidth, 
                                         const float* inputImage, float* outputImage, const float* kernel), 
               const string testName, const int iterations, vector<int>& kernels,  
               const int minKernelWidth, const int maxKernelWidth,
               const int imageStride, const int imageWidth, const int imageHeight, 
               float* inputImage, float* outputImage) { 
              
     //#if aligned
    cout << left << setw(50) << testName;
    for ( vector<int>::iterator it = kernels.begin(); it != kernels.end(); it++ ) {
        int kernelWidth = *it; 
        if(kernelWidth >= minKernelWidth && (kernelWidth <= maxKernelWidth || maxKernelWidth == 0)) {
            int kernelStride = calculateAlignedStride(kernelWidth, sizeof(float), ALIGMENT_BYTES);
            float* kernel = gaussianKernel2D(kernelWidth, 2);
            clear2DBuffer(outputImage, imageStride, imageHeight);
            vector<double> iter;
            for (int i = 0; i < iterations; i++) {
                clearCache();
//                prepareTestBuffers(imageStride, imageHeight, 
//                                   inputImage, outputImage);
                
                m_StopWatch.StartNew();    
                CALLGRIND_TOGGLE_COLLECT;    
                convolutionFunction(imageStride, imageWidth, imageHeight, 
                             kernelStride, kernelWidth, inputImage, 
                             outputImage, kernel);
                CALLGRIND_TOGGLE_COLLECT;
                CALLGRIND_STOP_INSTRUMENTATION;                               
                m_StopWatch.Stop();
                iter.push_back(m_StopWatch.GetElapsedTime()); 
            }  
            delete[] kernel;
            double mean = Mean(iter);
            double stdev = StDev(iter);
            cout << setw(10) << mean << setw(10) << setprecision(2) << naiveTimes[kernelWidth] / mean << 
                setprecision(6) << setw(10) << stdev << setw(10) << setprecision(1) << stdev / mean * 100 << 
                setprecision(6)  << flush; 
        }
        else {
            cout << setw(10) << "-" << setw(10) << "-" << setw(10) << "-" << setw(10) << "-" << flush; 
        }
        #ifdef DEBUGA
        printImage(imageWidth, imageHeight, imageStride, outputImage);
        #endif
        
    }
    cout << endl;

} 
 
void runSSETest(const string testName, const int iterations, vector<int>& kernels,  
               const int minKernelWidth, const int maxKernelWidth,
               const int imageStride, const int imageWidth, const int imageHeight, 
               float* inputImage, float* outputImage) { 
              
             
     //#if aligned
    cout << left << setw(50) << testName;
    for ( vector<int>::iterator it = kernels.begin(); it != kernels.end(); it++ ) {
        int kernelWidth = *it; 
        if(kernelWidth >= minKernelWidth && (kernelWidth <= maxKernelWidth || maxKernelWidth == 0)) {
            int kernelStride = calculateAlignedStride(kernelWidth, sizeof(float), ALIGMENT_BYTES);
            float* kernel = gaussianKernel2D(kernelWidth, 2);
            clear2DBuffer(outputImage, imageStride, imageHeight);
            vector<double> iter;
            for (int i = 0; i < iterations; i++) {
                clearCache();
//                prepareTestBuffers(imageStride, imageHeight, 
//                                   inputImage, outputImage);
                
                if(testName == "sse3Convolve"){
                    m_StopWatch.StartNew();    
                    CALLGRIND_TOGGLE_COLLECT;    
                    sse3Convolve(imageStride, imageWidth, imageHeight, 
                                 kernelStride, inputImage, 
                                 outputImage, kernel);
                    CALLGRIND_TOGGLE_COLLECT;
                    CALLGRIND_STOP_INSTRUMENTATION;                               
                    m_StopWatch.Stop();
                }  
                else if(testName == "sse3Convolve1"){ 
                    m_StopWatch.StartNew();    
                    sse3Convolve1(imageStride, imageWidth, imageHeight, 
                                 kernelStride, inputImage, 
                                 outputImage, kernel);                 
                    m_StopWatch.Stop();
                }
                else if(testName == "sse3ConvolveTest"){ 
                    m_StopWatch.StartNew();    
                    sse3ConvolveTest(imageStride, imageWidth, imageHeight, 
                                 kernelStride, inputImage, 
                                 outputImage, kernel);                 
                    m_StopWatch.Stop();
                }
                else if(testName == "sse3LbConvolve") {
                    m_StopWatch.StartNew();    
                    sse3LbConvolve(imageStride, imageWidth, imageHeight, 
                                 kernelStride, inputImage, 
                                 outputImage, kernel);
                    m_StopWatch.Stop();
                }
                else if(testName == "sse5Convolve") {
                    m_StopWatch.StartNew();    
                    sse5Convolve(imageStride, imageWidth, imageHeight, 
                                 kernelStride, inputImage, 
                                 outputImage, kernel);
                    m_StopWatch.Stop();
                }
                else if(testName == "sse5Convolve1") {
                    m_StopWatch.StartNew();    
                    sse5Convolve1(imageStride, imageWidth, imageHeight, 
                                 kernelStride, inputImage, 
                                 outputImage, kernel);
                    m_StopWatch.Stop();
                }
                else if(testName == "sse7Convolve") {
                    m_StopWatch.StartNew();    
                    sse7Convolve(imageStride, imageWidth, imageHeight, 
                                 kernelStride, inputImage, 
                                 outputImage, kernel);
                    m_StopWatch.Stop();
                }
                else if(testName == "sse7Convolve1") {
                    m_StopWatch.StartNew();    
                    sse7Convolve1(imageStride, imageWidth, imageHeight, 
                                 kernelStride, inputImage, 
                                 outputImage, kernel);
                    m_StopWatch.Stop();
                }
                else if(testName == "sse9Convolve") {
                    m_StopWatch.StartNew();    
                    sse9Convolve(imageStride, imageWidth, imageHeight, 
                                 kernelStride, inputImage, 
                                 outputImage, kernel);
                    m_StopWatch.Stop();
                }
                else if(testName == "sse9Convolve1") {
                    m_StopWatch.StartNew();    
                    sse9Convolve1(imageStride, imageWidth, imageHeight, 
                                 kernelStride, inputImage, 
                                 outputImage, kernel);
                    m_StopWatch.Stop();
                }
//                else if(testName == "sse11Convolve") {
//                    m_StopWatch.StartNew();    
//                    sse11Convolve(imageStride, imageWidth, imageHeight, 
//                                 kernelStride, inputImage, 
//                                 outputImage, kernel);
//                    m_StopWatch.Stop();
//                }
                iter.push_back(m_StopWatch.GetElapsedTime()); 
            }  
            delete[] kernel;
            double mean = Mean(iter);
            double stdev = StDev(iter);
            cout << setw(10) << mean << setw(10) << setprecision(2) << naiveTimes[kernelWidth] / mean << 
                setprecision(6) << setw(10) << stdev << setw(10) << setprecision(1) << stdev / mean * 100 << 
                setprecision(6)  << flush; 
        }
        else {
            cout << setw(10) << "-" << setw(10) << "-" << setw(10) << "-" << setw(10) << "-" << flush; 
        }
        #ifdef DEBUGA
        printImage(imageWidth, imageHeight, imageStride, outputImage);
        #endif
        
    }
    cout << endl;

}

void runLoopBlockConvolveTest(const string testName, const int iterations, vector<int>& kernels,  
               const int minKernelWidth, const int maxKernelWidth,
               const int imageStride, const int imageWidth, const int imageHeight, 
               float* inputImage, float* outputImage, const int xBlock, const int yBlock) { 

     //#if aligned
    cout << left << setw(50) << testName;
    for ( vector<int>::iterator it = kernels.begin(); it != kernels.end(); it++ ) {
        int kernelWidth = *it; 
        if(kernelWidth >= minKernelWidth && (kernelWidth <= maxKernelWidth || maxKernelWidth == 0)) {
            int kernelStride = calculateAlignedStride(kernelWidth, sizeof(float), ALIGMENT_BYTES);
            float* kernel = gaussianKernel2D(kernelWidth, 2);
            clear2DBuffer(outputImage, imageStride, imageHeight);
            vector<double> iter;
            for (int i = 0; i < iterations; i++) {
                clearCache();
//                prepareTestBuffers(imageStride, imageHeight, 
//                                   inputImage, outputImage);                
                if(testName == "loopBlock128x128Convolve") {
                    m_StopWatch.StartNew();    
                    loopBlockConvolve (imageStride, imageWidth, imageHeight, 
                                  kernelStride, kernelWidth, inputImage, 
                                  outputImage, kernel, xBlock, yBlock);                     
                    m_StopWatch.Stop();
                } 
                else if(testName == "loopBlock512x512Convolve") {
                    m_StopWatch.StartNew();    
                    loopBlockConvolve (imageStride, imageWidth, imageHeight, 
                                  kernelStride, kernelWidth, inputImage, 
                                  outputImage, kernel, xBlock, yBlock);                     
                    m_StopWatch.Stop();
                }
                else if(testName == "loopBlockLoopUnroll128x128Convolve") {
                    m_StopWatch.StartNew();     
                    loopBlockLoopUnrollConvolve (imageStride, imageWidth, imageHeight, 
                                  kernelStride, kernelWidth, inputImage, 
                                  outputImage, kernel, xBlock, yBlock);                     
                    m_StopWatch.Stop();
                }
                else if(testName == "loopBlockLoopUnroll512x512Convolve") {
                    m_StopWatch.StartNew();    
                    loopBlockLoopUnrollConvolve (imageStride, imageWidth, imageHeight, 
                                  kernelStride, kernelWidth, inputImage, 
                                  outputImage, kernel, xBlock, yBlock);                     
                    m_StopWatch.Stop();
                }
                else {
                    m_StopWatch.StartNew();    
                    loopBlockAlignedSSEConvolve (imageStride, imageWidth, imageHeight, 
                                  kernelStride, kernelWidth, inputImage, 
                                  outputImage, kernel, xBlock, yBlock);
                     
                    m_StopWatch.Stop();
                }
                iter.push_back(m_StopWatch.GetElapsedTime()); 
            }  
            delete[] kernel;
            double mean = Mean(iter);
            double stdev = StDev(iter);
            cout << setw(10) << mean << setw(10) << setprecision(2) << naiveTimes[kernelWidth] / mean << 
                setprecision(6) << setw(10) << stdev << setw(10) << setprecision(1) << stdev / mean * 100 << 
                setprecision(6)  << flush; 
        }
        else {
            cout << setw(10) << "-" << setw(10) << "-" << setw(10) << "-" << setw(10) << "-" << flush; 
        }
        #ifdef DEBUGA 
        printImage(imageWidth, imageHeight, imageStride, outputImage);
        #endif
        
    }
    cout << endl;
}


void runScTest(const string testName, const int iterations, vector<int>& kernels,  
               const int minKernelWidth, const int maxKernelWidth,
               const int imageStride, const int imageWidth, const int imageHeight, 
               float* inputImage, float* outputImage) { 
              
     //#if aligned
    cout << left << setw(50) << testName;
    for ( vector<int>::iterator it = kernels.begin(); it != kernels.end(); it++ ) {
        int kernelWidth = *it; 
        if(kernelWidth >= minKernelWidth && (kernelWidth <= maxKernelWidth || maxKernelWidth == 0)) {
            int kernelStride = calculateAlignedStride(kernelWidth, sizeof(float), ALIGMENT_BYTES);
            float* kernelX = gaussianKernel1D(kernelWidth, 2); 
            float* kernelY = gaussianKernel1D(kernelWidth, 2); 
            clear2DBuffer(outputImage, imageStride, imageHeight);
            vector<double> iter;
            for (int i = 0; i < iterations; i++) {
                clearCache();
//                prepareTestBuffers(imageStride, imageHeight, 
//                                   inputImage, outputImage);                
                if(testName == "separableConvolve"){
                    m_StopWatch.StartNew();    
                    separableConvolve (imageStride, imageWidth, imageHeight, 
                                   kernelWidth, inputImage, outputImage, kernelX, kernelY, true);
                    m_StopWatch.Stop();    
                }  
                else if(testName == "opSeparableConvolve") {
                    m_StopWatch.StartNew();    
                    opSeparableConvolve (imageStride, imageWidth, imageHeight, 
                        kernelWidth, inputImage, outputImage, kernelX, kernelY);
                    m_StopWatch.Stop();
                }
                else if(testName == "sse3SConvolve") {
                    m_StopWatch.StartNew();    
                    sse3SConvolve (imageStride, imageWidth, imageHeight, 
                        inputImage, outputImage, kernelX, kernelY);
                    m_StopWatch.Stop();
                }
                else if(testName == "sse5SConvolve") {
                    m_StopWatch.StartNew();    
                    sse5SConvolve (imageStride, imageWidth, imageHeight, 
                        inputImage, outputImage, kernelX, kernelY);
                    m_StopWatch.Stop();
                }
                else if(testName == "sse7SConvolve") {
                    m_StopWatch.StartNew();    
                    sse7SConvolve (imageStride, imageWidth, imageHeight, 
                        inputImage, outputImage, kernelX, kernelY);
                    m_StopWatch.Stop();
                }
                else if(testName == "sse9SConvolve") {
                    m_StopWatch.StartNew();    
                    sse9SConvolve (imageStride, imageWidth, imageHeight, 
                        inputImage, outputImage, kernelX, kernelY);
                    m_StopWatch.Stop();
                }
                else if(testName == "sse5GaussianSConvolve") {
                    m_StopWatch.StartNew();    
                    sse5GaussianSConvolve (imageStride, imageWidth, imageHeight, 
                                    inputImage, outputImage, kernelX);
                    m_StopWatch.Stop();
                }
                else if(testName == "sse7GaussianSConvolve") {
                    m_StopWatch.StartNew();    
                    sse7GaussianSConvolve (imageStride, imageWidth, imageHeight, 
                                    inputImage, outputImage, kernelX);
                    m_StopWatch.Stop();
                }
                else if(testName == "sse9GaussianSConvolve") {
                    m_StopWatch.StartNew();    
                    sse9GaussianSConvolve (imageStride, imageWidth, imageHeight, 
                                    inputImage, outputImage, kernelX);
                    m_StopWatch.Stop();
                }
                else if(testName == "sseSConvolve") {
                    m_StopWatch.StartNew();    
                    sseSConvolve (imageStride, imageWidth, imageHeight, kernelWidth,
                        inputImage, outputImage, kernelX, kernelY);
                    m_StopWatch.Stop();
                }
                iter.push_back(m_StopWatch.GetElapsedTime()); 
            }  
            delete[] kernelX;
            delete[] kernelY;
            double mean = Mean(iter);
            double stdev = StDev(iter);
            if (testName == "separableConvolve") {
                separableTimes[kernelWidth] = mean;
                cout << setw(10) << mean << setw(10) << setprecision(2) << naiveTimes[kernelWidth] / mean << 
                    setprecision(6) << setw(10) << stdev << setw(10) << setprecision(1) << stdev / mean * 100 << 
                    setprecision(6)  << flush;                 
            }
            else {
                cout << setw(10) << mean << setw(10) << setprecision(2) << separableTimes[kernelWidth] / mean << 
                    setprecision(6) << setw(10) << stdev << setw(10) << setprecision(1) << stdev / mean * 100 << 
                    setprecision(6)  << flush; 
            }
        }
        else {
            cout << setw(10) << "-" << setw(10) << "-" << setw(10) << "-" << setw(10) << "-" << flush; 
        }
        #ifdef DEBUGA
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
            sseUnalignedConvolve(imageStride, imageWidth, imageHeight, 
                            kernelStride, kernelWidth, inputImage, 
                            outputImage, kernel); 
            if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
                stringstream f;
                f << "sseUnalignedConvolve fail!" << endl;
                f << s.str();
                assertFailList.push_back(f.str());
            }
 
            clear2DBuffer(outputImage, imageStride, imageHeight);
            sseUnaligned4Convolve(imageStride, imageWidth, imageHeight, 
                            kernelStride, kernelWidth, inputImage, 
                            outputImage, kernel); 
            if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
                stringstream f;
                f << "sseUnaligned4Convolve fail!" << endl;
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
                prefetch64Convolve(imageStride, imageWidth, imageHeight, 
                                kernelStride, kernelWidth, inputImage, 
                                outputImage, kernel); 
                if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
                    stringstream f;
                    f << "prefetch64Convolve fail!" << endl;
                    f << s.str();
                    assertFailList.push_back(f.str());
                }
            }
            
            if(imageWidth > 128 &&  imageHeight > 128) {                        
                clear2DBuffer(outputImage, imageStride, imageHeight);
                prefetch128Convolve(imageStride, imageWidth, imageHeight, 
                                kernelStride, kernelWidth, inputImage, 
                                outputImage, kernel); 
                if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
                    stringstream f;
                    f << "prefetch128Convolve fail!" << endl;
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
     
            if(kernelWidth == 3) {                        
                clear2DBuffer(outputImage, imageStride, imageHeight);
                sse3Convolve(imageStride, imageWidth, imageHeight, 
                            kernelStride, inputImage, 
                            outputImage, kernel);
                if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
                    stringstream f;
                    f << "sse3Convolve fail!" << endl;
                    f << s.str();
                    assertFailList.push_back(f.str());
                }
            }
                        
            if(kernelWidth == 3) {                        
                clear2DBuffer(outputImage, imageStride, imageHeight);
                sse3ConvolveTest(imageStride, imageWidth, imageHeight, 
                            kernelStride, inputImage, 
                            outputImage, kernel);                 
                if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
                    stringstream f;
                    f << "sse3ConvolveTest fail!" << endl;
                    f << s.str();
                    assertFailList.push_back(f.str());
                }
            }     
            
            if(kernelWidth == 3) {                        
                clear2DBuffer(outputImage, imageStride, imageHeight);
                sse3LbConvolve(imageStride, imageWidth, imageHeight, 
                            kernelStride, inputImage, 
                            outputImage, kernel);
                if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
                    stringstream f;
                    f << "sse3LbConvolve fail!" << endl;
                    f << s.str();
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
                    assertFailList.push_back(f.str());
                }
            }
                        
//            if(kernelWidth == 11) {                        
//                clear2DBuffer(outputImage, imageStride, imageHeight);
//                sse11Convolve(imageStride, imageWidth, imageHeight, 
//                            kernelStride, inputImage, 
//                            outputImage, kernel);
//                if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
//                    stringstream f;
//                    f << "sse11Convolve fail!" << endl;
//                    f << s.str();
//                    assertFailList.push_back(f.str());
//                }
//            }           
            
            clear2DBuffer(outputImage, imageStride, imageHeight);
            separableConvolve (imageStride, imageWidth, imageHeight, 
                        kernelWidth, inputImage, outputImage, kernelX, kernelY, true);
            if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
                stringstream f;
                f << "separableConvolve fail!" << endl;
                f << s.str();
                assertFailList.push_back(f.str());
            }
                        
            clear2DBuffer(outputImage, imageStride, imageHeight);
            sseSConvolve (imageStride, imageWidth, imageHeight, 
                   kernelWidth, inputImage, outputImage, kernelX, kernelY);
            if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
                stringstream f;
                f << "sseSConvolve fail!" << endl;
                f << s.str();
                assertFailList.push_back(f.str());
            }
            
            if(kernelWidth == 3) {
                clear2DBuffer(outputImage, imageStride, imageHeight);
                sse3SConvolve (imageStride, imageWidth, imageHeight, 
                            inputImage, outputImage, kernelX, kernelY);
                if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
                    stringstream f;
                    f << "sse3SConvolve fail!" << endl;
                    f << s.str();
                    assertFailList.push_back(f.str());
                }
            }           
            
            if(kernelWidth == 5) {
                clear2DBuffer(outputImage, imageStride, imageHeight);
                sse5SConvolve (imageStride, imageWidth, imageHeight, 
                            inputImage, outputImage, kernelX, kernelY);
                if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
                    stringstream f;
                    f << "sse5SConvolve fail!" << endl;
                    f << s.str();
                    assertFailList.push_back(f.str());
                }
            }
            
            if(kernelWidth == 7) {
                clear2DBuffer(outputImage, imageStride, imageHeight);
                sse7SConvolve (imageStride, imageWidth, imageHeight, 
                            inputImage, outputImage, kernelX, kernelY);
                if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
                    stringstream f;
                    f << "sse7SConvolve fail!" << endl;
                    f << s.str();
                    assertFailList.push_back(f.str());
                }
            }
            
            if(kernelWidth == 9) {                        
                clear2DBuffer(outputImage, imageStride, imageHeight);
                sse9SConvolve (imageStride, imageWidth, imageHeight, 
                            inputImage, outputImage, kernelX, kernelY);
                if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
                    stringstream f;
                    f << "sse9SConvolve fail!" << endl;
                    f << s.str();
                    assertFailList.push_back(f.str());
                }
            }    
            

//            for(int y = 16; y <= 512; y += 16) {
//                for(int x = 16; x <= 512; x += 16) {
//                    clear2DBuffer(outputImage, imageStride, imageHeight);
//                    loopBlockConvolve(imageStride, imageWidth, imageHeight, 
//                                 kernelStride, kernelWidth, inputImage, 
//                                 outputImage, kernel, x, y);
//                    if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
//                        stringstream f;
//                        f << "loopBlockConvolve fail!" << endl;
//                        f << s.str();
//                        assertFailList.push_back(f.str());
//                    }
//                    clear2DBuffer(outputImage, imageStride, imageHeight);
//                    loopBlockLoopUnrollConvolve(imageStride, imageWidth, imageHeight, 
//                                 kernelStride, kernelWidth, inputImage, 
//                                 outputImage, kernel, x, y);
//                    if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
//                        stringstream f;
//                        f << "loopBlockLoopUnrollConvolve fail!" << endl;
//                        f << s.str();
//                        assertFailList.push_back(f.str());
//                    }
//                    clear2DBuffer(outputImage, imageStride, imageHeight);
//                    loopBlockAlignedSSEConvolve(imageStride, imageWidth, imageHeight, 
//                                 kernelStride, kernelWidth, inputImage, 
//                                 outputImage, kernel, x, y);
//                                 
//                    if(!assertConvolution(naiveOutputImage, outputImage, imageWidth, imageHeight, imageWidth, imageStride, kernelWidth)) {
//                        stringstream f;
//                        f << "loopBlockAlignedSSEConvolve fail!" << endl;
//                        f << s.str();
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
    int Width;
    int Height;
    ImageSize(int w, int h) {
        Width = w;
        Height = h;
    }
    string ToString() {
        return "[" + lexical_cast<std::string>(Width) + ", " + lexical_cast<std::string>(Height) + "]";
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
            const int imageWidth = is.Width; 
            const int imageHeight = is.Height; 
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
                    x == 512 ? outFile << x : outFile << x << ",";
                }
                outFile << endl;
                for(int y = 16; y <= 512; y += 16) {
                    outFile <<  y << ","; 
                    for(int x = 16; x <= 512; x += 16) {
                        float m = 0;
                        for (int i = 0; i < 10; i++) {
                            m_StopWatch.StartNew();    
                            loopBlockAlignedSSEConvolve(imageStride, imageWidth, imageHeight, 
                                         kernelStride, kernelWidth, inputImage, 
                                         outputImage, kernel, x, y);
                            m_StopWatch.Stop();
                            m += m_StopWatch.GetElapsedTime();     
                        }
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


void naiveConvolveTest( const int iterations, vector<int>& kernels, 
                        const int imageWidth, const int imageHeight, 
                        const int kernelWidth, 
                        const float* inputImage, const float* kernel) {
    //#if NAIVE
    cout << left << setw(50) << "naiveConvolve";
    float* naiveInputImage = new float[imageWidth * imageHeight];
    float* naiveOutputImage = new float[imageWidth * imageHeight];
    for ( vector<int>::iterator it = kernels.begin(); it != kernels.end(); it++ ) {
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
     
         vector<double> iter;
         for (int i = 0; i < iterations; i++) {
             m_StopWatch.StartNew();
             CALLGRIND_TOGGLE_COLLECT;    
             naiveConvolve(imageWidth, imageWidth, imageHeight, 
                             kernelWidth, kernelWidth, naiveInputImage, 
                             naiveOutputImage, naiveKernel);
             CALLGRIND_TOGGLE_COLLECT;
             CALLGRIND_STOP_INSTRUMENTATION;                               
             m_StopWatch.Stop();
              iter.push_back(m_StopWatch.GetElapsedTime()); 
         }
        delete[] kernel;
        double mean = Mean(iter);
        double stdev = StDev(iter);
        cout << setw(10) << mean << setw(10) << setprecision(2) << 1.0 << setprecision(6) << setw(10) << stdev << setw(10) << setprecision(1) << stdev / mean * 100 << setprecision(6)  << flush; 
        naiveTimes[kernelWidth]= mean;
        
//        cout.precision(2); 
//        printImage(imageWidth, imageHeight, imageWidth, naiveOutputImage);
//        cout.precision(4); 
        
    }
    cout << endl;
  
    delete[] naiveOutputImage;
    delete[] naiveInputImage;
 
}

static void prepareTestBuffers (const int imageStride, const int imageHeight, 
                                float* inputImage, float* outputImage) {
                                 
    /* initialize random seed: */
    srand ( time(NULL) );
    for (int i = 0; i < imageStride * imageHeight; i++) {
         inputImage[i] =  rand() * 255;
    }       
}

static void clearCache() {

    int cacheSize = sysconf (_SC_LEVEL1_DCACHE_SIZE) / 4;
    float* cacheIn;
    float* cacheOut;
    if (cacheSize) {
        cacheIn = new float[cacheSize];
        cacheOut = new float[cacheSize];
        clear2DBuffer (cacheIn, cacheSize, 1);
        for (int i = 0; i < cacheSize; i++) {
             cacheOut[i] = cacheIn[i];
        }
        delete [] cacheIn;
        delete [] cacheOut;
    } 
    
    cacheSize = sysconf (_SC_LEVEL2_CACHE_SIZE) / 4;
    if (cacheSize) {
        cacheIn = new float[cacheSize];
        cacheOut = new float[cacheSize];
        clear2DBuffer (cacheIn, cacheSize, 1);
        for (int i = 0; i < cacheSize; i++) {
             cacheOut[i] = cacheIn[i];
        }
        delete [] cacheIn;
        delete [] cacheOut;
    }
  
    cacheSize = sysconf (_SC_LEVEL3_CACHE_SIZE) / 4;
    if (cacheSize) {
        cacheIn = new float[cacheSize];
        cacheOut = new float[cacheSize];
        clear2DBuffer (cacheIn, cacheSize, 1);
        for (int i = 0; i < cacheSize; i++) {
             cacheOut[i] = cacheIn[i];
        }
        delete [] cacheIn;
        delete [] cacheOut;
    }
 
}

int main (int argc, char *argv[])
{
    #ifndef __SSE4_1__
        cout << "Running in AMD architecture..." << endl;
    #endif
    
    /* initialize random seed: */
    srand ( time(NULL) );
    cout.setf(ios::fixed);
    cout.precision(4); 
    
    int optind=1;
    string configFile;
    // decode arguments
    while ((optind < argc) && (argv[optind][0]=='-')) { 
        string sw = argv[optind];
        if (sw=="-c") {
            configFile = argv[optind + 1];
        }
        else if (sw=="-a") {
            assertTest(); 
            return 0;
        }
        else if (sw=="-lb") {
            loopBlockAlignedSSEConvolveTest();  
            return 0;
        }
        optind++;
    }    
    
    string line;
    ifstream confFile (configFile.c_str());
    map<string,string> config;
    
    if (confFile.is_open()) {
        while ( confFile.good() )
        {
            getline (confFile, line);
            int eqPos = line.find("=");
            string key = line.substr (0, eqPos);
            string value = line.substr (eqPos + 1, line.length() - eqPos + 1);
            config[key] = value;
        }
            confFile.close();
        }
    else {
        cout << "Unable to open test.cfg file" << flush << endl; 
        return 1;
    }
    
    vector<string> strings = split(config["kernels"], ',');  
    vector<int> kernels;
    
    std::transform(strings.begin(), strings.end(), 
               std::back_inserter(kernels), 
               lexical_cast<int, std::string>); // Note the two template arguments!    
    
    vector<string> algs = split(config["algs"], ',');  
    
    
    vector<ImageSize> images;
    
    foreach (string s, split(config["images"], ',')) {
        vector<string> simg = split(s, 'x');
        images.push_back(ImageSize(lexical_cast<int>(simg[0]),
                                   lexical_cast<int>(simg[1])));
    }
    
#ifdef _OPENMP     
    omp_set_num_threads(lexical_cast<int>(config["maxthreads"]));
#endif
    
    foreach (ImageSize image, images) {
     
        int imageWidth = image.Width;
        int imageHeight = image.Height;
        
        string line = string(50 + kernels.size() * 40, '-');
        cout << line << endl; 
        cout << "image size " << image.ToString() << " - max threads " << config["maxthreads"] << endl;
        cout << line << endl;
        cout << left << setw(50) << "algorithm/kernel size";
        foreach (int i, kernels) {
            cout << setw(10) << lexical_cast<string>(i) << setw(10) << "spdup" << setw(10) << "stdev" << setw(10) << "%"; 
        }
        cout << endl;
//        cout << string(40 + kernels.size() * 32, '-') << endl;
        
       
        int iterations = lexical_cast<int>(config["iterations"]);
        
        float* inputImage = allocateFloatAlignedBuffer(imageWidth, imageHeight);
        float* outputImage = allocateFloatAlignedBuffer(imageWidth, imageHeight);
        int imageStride  = calculateAlignedStride(imageWidth, sizeof(float), ALIGMENT_BYTES);
        
        int kernelWidth = 0;
        int kernelStride = 0;
        float* kernel;
        
        CALLGRIND_START_INSTRUMENTATION;
         
        prepareTestBuffers(imageStride, imageHeight, 
                           inputImage, outputImage);                
         
        if(find(algs.begin(), algs.end(), "naiveConvolve") != algs.end())
            naiveConvolveTest (iterations, kernels, imageWidth, imageHeight, 
                               kernelWidth, inputImage, kernel);
        
        if(find(algs.begin(), algs.end(), "alignedConvolve") != algs.end())
            run2DTest (alignedConvolve, "alignedConvolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
                       inputImage, outputImage);
     
        if(find(algs.begin(), algs.end(), "sseUnalignedConvolve") != algs.end())
            run2DTest (sseUnalignedConvolve, "sseUnalignedConvolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
                     inputImage, outputImage) ; 
                 
        if(find(algs.begin(), algs.end(), "sseUnaligned4Convolve") != algs.end())
            run2DTest (sseUnalignedConvolve, "sseUnaligned4Convolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
                     inputImage, outputImage) ;
                 
        if(find(algs.begin(), algs.end(), "loopUnrollConvolve") != algs.end())
            run2DTest (loopUnrollConvolve, "loopUnrollConvolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
                     inputImage, outputImage);
                 
        if(find(algs.begin(), algs.end(), "prefetch64Convolve") != algs.end())
            run2DTest (prefetch64Convolve, "prefetch64Convolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
                     inputImage, outputImage);
    
        if(find(algs.begin(), algs.end(), "prefetch128Convolve") != algs.end())
            run2DTest (prefetch128Convolve, "prefetch128Convolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
                     inputImage, outputImage);
                     
        if(find(algs.begin(), algs.end(), "sse3Convolve") != algs.end()) {
            runSSETest ("sse3Convolve", iterations, kernels, 3, 3, imageStride, imageWidth, imageHeight, 
                     inputImage, outputImage);
//            cout.precision(2); 
//            printImage(imageWidth, imageHeight, imageStride, outputImage);
//            cout.precision(4); 
                     
        }
    
        if(find(algs.begin(), algs.end(), "sse3Convolve1") != algs.end()) {
            runSSETest ("sse3Convolve1", iterations, kernels, 3, 3, imageStride, imageWidth, imageHeight, 
                     inputImage, outputImage);
//            cout.precision(2); 
//            printImage(imageWidth, imageHeight, imageStride, outputImage);
//            cout.precision(4); 
                     
        }
    
        if(find(algs.begin(), algs.end(), "sse3ConvolveTest") != algs.end())
            runSSETest ("sse3ConvolveTest", iterations, kernels, 3, 3, imageStride, imageWidth, imageHeight, 
                     inputImage, outputImage);
    
        if(find(algs.begin(), algs.end(), "sse3LbConvolve") != algs.end())
            runSSETest ("sse3LbConvolve", iterations, kernels, 3, 3, imageStride, imageWidth, imageHeight, 
                     inputImage, outputImage);
    
        if(find(algs.begin(), algs.end(), "sse5Convolve") != algs.end()) {
            runSSETest ("sse5Convolve", iterations, kernels, 5, 5, imageStride, imageWidth, imageHeight, 
                     inputImage, outputImage);
//            cout.precision(2); 
//            printImage(imageWidth, imageHeight, imageStride, outputImage);
//            cout.precision(4); 
        }
                     
        if(find(algs.begin(), algs.end(), "sse5Convolve1") != algs.end()) {
            runSSETest ("sse5Convolve1", iterations, kernels, 5, 5, imageStride, imageWidth, imageHeight, 
                     inputImage, outputImage);
//            cout.precision(2); 
//            printImage(imageWidth, imageHeight, imageStride, outputImage);
//            cout.precision(4); 
        }
                     
    
        if(find(algs.begin(), algs.end(), "sse7Convolve") != algs.end())
            runSSETest ("sse7Convolve", iterations, kernels, 7, 7, imageStride, imageWidth, imageHeight, 
                     inputImage, outputImage);
    
        if(find(algs.begin(), algs.end(), "sse7Convolve1") != algs.end())
            runSSETest ("sse7Convolve1", iterations, kernels, 7, 7, imageStride, imageWidth, imageHeight, 
                     inputImage, outputImage);
    
        if(find(algs.begin(), algs.end(), "sse9Convolve") != algs.end()) {
            runSSETest ("sse9Convolve", iterations, kernels, 9, 9, imageStride, imageWidth, imageHeight, 
                     inputImage, outputImage);
//            cout.precision(2); 
//            printImage(imageWidth, imageHeight, imageStride, outputImage);
//            cout.precision(4); 
                     
        }
    
        if(find(algs.begin(), algs.end(), "sse9Convolve1") != algs.end())
            runSSETest ("sse9Convolve1", iterations, kernels, 9, 9, imageStride, imageWidth, imageHeight, 
                     inputImage, outputImage);
    
//        if(find(algs.begin(), algs.end(), "sse11Convolve") != algs.end())
//            runSSETest ("sse11Convolve", iterations, kernels, 11, 11, imageStride, imageWidth, imageHeight, 
//                     inputImage, outputImage);
    
        if(find(algs.begin(), algs.end(), "sseNoReuse1Convolve") != algs.end())
            run2DTest (sseNoReuse1Convolve, "sseNoReuse1Convolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
                     inputImage, outputImage);
    
        if(find(algs.begin(), algs.end(), "sseNoReuse2Convolve") != algs.end())
            run2DTest (sseNoReuse2Convolve, "sseNoReuse2Convolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
                     inputImage, outputImage);
    
        if(find(algs.begin(), algs.end(), "sseNoReuse3Convolve") != algs.end())
            run2DTest (sseNoReuse3Convolve, "sseNoReuse3Convolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
                     inputImage, outputImage);
    
        if(find(algs.begin(), algs.end(), "sseNoReuse4Convolve") != algs.end())
            run2DTest (sseNoReuse4Convolve, "sseNoReuse4Convolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
                     inputImage, outputImage);
    
        if(find(algs.begin(), algs.end(), "sseNoReuse5Convolve") != algs.end())
            run2DTest (sseNoReuse5Convolve, "sseNoReuse5Convolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
                     inputImage, outputImage); 
    
        if(find(algs.begin(), algs.end(), "sseNoReuse6Convolve") != algs.end()) 
            run2DTest (sseNoReuse6Convolve, "sseNoReuse6Convolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
                     inputImage, outputImage);
    
        if(find(algs.begin(), algs.end(), "sseNoReuse7Convolve") != algs.end())
            run2DTest (sseNoReuse7Convolve, "sseNoReuse7Convolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
                     inputImage, outputImage);
    
        if(find(algs.begin(), algs.end(), "sseReuse1Convolve") != algs.end())
            run2DTest (sseReuse1Convolve, "sseReuse1Convolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
                     inputImage, outputImage);
    
        if(find(algs.begin(), algs.end(), "sseReuse2Convolve") != algs.end())
            run2DTest (sseReuse2Convolve, "sseReuse2Convolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
                     inputImage, outputImage);
    
        if(find(algs.begin(), algs.end(), "sseReuse3Convolve") != algs.end())
            run2DTest (sseReuse3Convolve, "sseReuse3Convolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
                     inputImage, outputImage);
    
        if(find(algs.begin(), algs.end(), "sseReuse4Convolve") != algs.end())
            run2DTest (sseReuse4Convolve, "sseReuse4Convolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
                     inputImage, outputImage);
    
        if(find(algs.begin(), algs.end(), "sseReuse5Convolve") != algs.end())
            run2DTest (sseReuse5Convolve, "sseReuse5Convolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
                     inputImage, outputImage);
    
        if(find(algs.begin(), algs.end(), "sseReuse6Convolve") != algs.end())
            run2DTest (sseReuse6Convolve, "sseReuse6Convolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
                     inputImage, outputImage);
    
        if(find(algs.begin(), algs.end(), "sseReuse7Convolve") != algs.end())
            run2DTest (sseReuse7Convolve, "sseReuse7Convolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
                     inputImage, outputImage);
                     
        if(find(algs.begin(), algs.end(), "opConvolve") != algs.end())
            run2DTest (opConvolve, "opConvolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
                     inputImage, outputImage);
    
        if(find(algs.begin(), algs.end(), "loopBlock128x128Convolve") != algs.end())
            runLoopBlockConvolveTest("loopBlock128x128Convolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
                     inputImage, outputImage, 128, 128);
    
        if(find(algs.begin(), algs.end(), "loopBlock128x128Convolve") != algs.end())
            runLoopBlockConvolveTest("loopBlock512x512Convolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
                     inputImage, outputImage, 128, 128);
    
        if(find(algs.begin(), algs.end(), "loopBlockLoopUnroll128x128Convolve") != algs.end())
            runLoopBlockConvolveTest("loopBlockLoopUnroll128x128Convolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
                     inputImage, outputImage, 128, 128);
    
        if(find(algs.begin(), algs.end(), "loopBlockLoopUnroll128x128Convolve") != algs.end())
            runLoopBlockConvolveTest("loopBlockLoopUnroll512x512Convolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
                     inputImage, outputImage, 512, 512);
                 
        if(find(algs.begin(), algs.end(), "loopBlock128x128AlignedSSEConvolve") != algs.end())
            runLoopBlockConvolveTest("loopBlock128x128AlignedSSEConvolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
                     inputImage, outputImage, 128, 128);
    
        if(find(algs.begin(), algs.end(), "loopBlock512x512AlignedSSEConvolve") != algs.end())
            runLoopBlockConvolveTest("loopBlock512x512AlignedSSEConvolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
                     inputImage, outputImage, 512, 512);
    
        if(find(algs.begin(), algs.end(), "separableConvolve") != algs.end())
            runScTest ("separableConvolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
                     inputImage, outputImage);
    
        if(find(algs.begin(), algs.end(), "sseSConvolve") != algs.end())
            runScTest ("sseSConvolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
                     inputImage, outputImage);

         if(find(algs.begin(), algs.end(), "sse3SConvolve") != algs.end())
            runScTest ("sse3SConvolve", iterations, kernels, 3, 3, imageStride, imageWidth, imageHeight, 
                     inputImage, outputImage);
    
        if(find(algs.begin(), algs.end(), "sse5SConvolve") != algs.end())
            runScTest ("sse5SConvolve", iterations, kernels, 5, 5, imageStride, imageWidth, imageHeight, 
                     inputImage, outputImage);
    
        if(find(algs.begin(), algs.end(), "sse7SConvolve") != algs.end())
            runScTest ("sse7SConvolve", iterations, kernels, 7, 7, imageStride, imageWidth, imageHeight, 
                     inputImage, outputImage);
    
        if(find(algs.begin(), algs.end(), "sse9SConvolve") != algs.end())
            runScTest ("sse9SConvolve", iterations, kernels, 9, 9, imageStride, imageWidth, imageHeight, 
                     inputImage, outputImage);
    
        if(find(algs.begin(), algs.end(), "opSeparableConvolve") != algs.end())
            runScTest ("opSeparableConvolve", iterations, kernels, 2, 0, imageStride, imageWidth, imageHeight, 
                     inputImage, outputImage);
    
        if(find(algs.begin(), algs.end(), "sse5GaussianSConvolve") != algs.end())
            runScTest ("sse5GaussianSConvolve", iterations, kernels, 5, 5, imageStride, imageWidth, imageHeight, 
                     inputImage, outputImage);
    
        if(find(algs.begin(), algs.end(), "sse7GaussianSConvolve") != algs.end())
            runScTest ("sse7GaussianSConvolve", iterations, kernels, 7, 7, imageStride, imageWidth, imageHeight, 
                     inputImage, outputImage);
    
        if(find(algs.begin(), algs.end(), "sse9GaussianSConvolve") != algs.end())
            runScTest ("sse9GaussianSConvolve", iterations, kernels, 9, 9, imageStride, imageWidth, imageHeight, 
                     inputImage, outputImage);
    
        delete[] inputImage;
        delete[] outputImage;
    cout << line << endl;
    cout << endl;
    cout << endl;
    }
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
                
                if ((b == 0 && a != 0) ||  1 - (a / b) > 0.005 ) {
//                    cout << "w " << imageWidth << "  h " << imageHeight << "  kw " << kernelWidth << endl; 
//                    cout << "a,b " << a << " " << b << endl;
//                    cout << "abs " << 1 - (a / b) << endl;
//                    cout << "image" << endl;
//                    printImage(imageWidth, imageHeight, convolveStride, kernelWidth, convolveOutput);
//                    cout << "end image" << endl;
                    return false;
                } 
            }
        }   
    return true;
}

float* gaussianKernel2D(const int kernelWidth, const float sigma) {
 
    
    const int radius = kernelWidth / 2;
 
    float* kernel = allocateFloatAlignedBuffer(kernelWidth, kernelWidth);
    int kernelStride = calculateAlignedStride(kernelWidth, sizeof(float), ALIGMENT_BYTES);

     for (int i = 0; i < kernelStride * kernelWidth; i++) {
         kernel[i] = 0;
    }       
    
    #ifdef DEBUGA 
    cout << endl;
    cout << "Gaussian2D kernel" << endl;
    cout << "kernelWidth " << kernelWidth << endl;
    #endif
    
    //http://homepages.inf.ed.ac.uk/rbf/HIPR2/gsmooth.htm
    for(int y = -radius; y < radius + 1; ++y) {
        for(int x = -radius; x < radius + 1; ++x) {
            float value = exp( (pow(x,2) + pow(y,2)) / (-2 * pow(sigma, 2))) / (2 * M_PI * pow(sigma, 2));       
            //kernel[(y + radius) * kernelStride + (x + radius)] = value;
            kernel[(y + radius) * kernelStride + (x + radius)] = x + 2;
            #ifdef DEBUGA 
            cout << value << " ";
            #endif
        }
        #ifdef DEBUGA 
        cout << endl;
        #endif
    }
    return kernel;
}

float* gaussianKernel1D(const int kernelWidth, const float sigma) {
 
    
    const int radius = kernelWidth / 2;
    float* kernel __attribute__ ((aligned(ALIGMENT_BYTES))) = new float[kernelWidth + 4 - (kernelWidth % 4)]; 
    for (int i = 0; i < kernelWidth + 4 - (kernelWidth % 4); i++) {
         kernel[i] = 0;
    }       
    
    //http://homepages.inf.ed.ac.uk/rbf/HIPR2/gsmooth.htm
    for(int x = -radius; x < radius + 1; ++x) {
            float value = exp ( pow ( x, 2 ) / ( -2 * pow ( sigma, 2 ) ) ) / ( sqrt( 2 * M_PI ) * sigma );
            kernel[x + radius] = value;
    }
    return kernel;
}
