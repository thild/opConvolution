// 
// opConvutionFilter.h
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

#ifndef _OPCONVUTIONFILTER_H
#define _OPCONVUTIONFILTER_H

#include <smmintrin.h>
#include <emmintrin.h>
#include <xmmintrin.h>

#include <iostream>

//#define _USE_MATH_DEFINES
#define ALIGMENT_BYTES 64 
#define CACHE_LINE_SIZE 64


//debug helpers

//#define TRACE
#ifdef TRACE
    #define PRINT_TRACE(x) \
        cout << #x << ":\t" << x << endl;
    #define PRINT_VECTOR_TRACE(x) \
        cout << #x << " { " << __builtin_ia32_vec_ext_v4sf(x, 0) << " " \
                     << __builtin_ia32_vec_ext_v4sf(x, 1) << " " \
                     << __builtin_ia32_vec_ext_v4sf(x, 2) << " " \
                     << __builtin_ia32_vec_ext_v4sf(x, 3) << " } " << endl;        
#else
    #define PRINT_TRACE(x)
    #define PRINT_VECTOR_TRACE(x) 
#endif                      

//#define DEBUG
#ifndef DEBUG
 #define PRINT_IMAGE_TO_FILE(file, image, width, height, s)
    #define PRINT_IMAGE(width, height, s, kw, out)
    #define PRINT_LABEL(x)
    #define PRINT(x)
    #define PRINT_INLINE(x)
    #define PRINT_LINE()
    #define PRINT_VECTOR(x)
    #define PRINT_POSITION(x, y)
#else

 #define PRINT_IMAGE_TO_FILE(file, width, height, s, out) \
     printImageToFile(file, width, height, s, out);
    #define PRINT_IMAGE(width, height, s, kw, out) \
     printImage(width, height, s, kw, out)
 #define PRINT_VECTOR(x) \
 cout << #x << " { " << __builtin_ia32_vec_ext_v4sf(x, 0) << " " \
                     << __builtin_ia32_vec_ext_v4sf(x, 1) << " " \
                     << __builtin_ia32_vec_ext_v4sf(x, 2) << " " \
                     << __builtin_ia32_vec_ext_v4sf(x, 3) << " } " << endl;
 #define PRINT_POSITION(x, y) \
     cout << " { " << __builtin_ia32_vec_ext_v4sf(x, y) << " } ";
    #define PRINT_LABEL(x) \
     cout << endl << " ### " << x << " ### " << endl;
    #define PRINT(x) \
     cout << #x << ":\t" << x << endl;
    #define PRINT_INLINE(x) \
     cout << x << " ";
    #define PRINT_LINE() \
     cout << endl;   
 
#endif                       

 
  
   
//SSE instruction helpers    
                          
#define ROTATE_LEFT(vector) \
    vector = _mm_shuffle_ps(vector, vector, _MM_SHUFFLE(0,3,2,1)); PRINT_VECTOR(vector); 

#define ROTATE_RIGHT(vector) \
    vector = _mm_shuffle_ps(vector, vector, _MM_SHUFFLE(2, 1, 0, 3)); PRINT_VECTOR(vector); 
                     
#define ROTATE_RIGHT_BLEND(vector1, vector2) \
    vector1 = _mm_shuffle_ps(vector1, vector1, _MM_SHUFFLE(2, 1, 0, 3)); PRINT_VECTOR(vector1); \
    vector2 = _mm_blend_ps(vector2, vector1, 1); PRINT_VECTOR(vector2); 
                     
#define BLEND_ROTATE_LEFT(vector0, vector1) \
    vector0 = _mm_blend_ps(vector0, vector1, 1); PRINT_VECTOR(vector0); \
    ROTATE_LEFT(vector0);

#define BLEND_ROTATE1_LEFT(vector0, vector1) \
    BLEND_ROTATE_LEFT(vector0, vector1) \
    ROTATE_LEFT(vector1)

#define BLEND_ROTATE2_LEFT(vector0, vector1, vector2) \
    BLEND_ROTATE_LEFT(vector0, vector1) \
    BLEND_ROTATE_LEFT(vector1, vector2) \
    ROTATE_LEFT(vector2)

#define BLEND_ROTATE3_LEFT(vector0, vector1, vector2, vector3) \
    BLEND_ROTATE_LEFT(vector0, vector1) \
    BLEND_ROTATE_LEFT(vector1, vector2) \
    BLEND_ROTATE_LEFT(vector2, vector3) \
    ROTATE_LEFT(vector3)

#define BLEND_ROTATE4_LEFT(vector0, vector1, vector2, vector3, vector4) \
    BLEND_ROTATE_LEFT(vector0, vector1) \
    BLEND_ROTATE_LEFT(vector1, vector2) \
    BLEND_ROTATE_LEFT(vector2, vector3) \
    BLEND_ROTATE_LEFT(vector3, vector4) \
    ROTATE_LEFT(vector4)

#define BLEND_ROTATE5_LEFT(vector0, vector1, vector2, vector3, vector4, vector5) \
    BLEND_ROTATE_LEFT(vector0, vector1) \
    BLEND_ROTATE_LEFT(vector1, vector2) \
    BLEND_ROTATE_LEFT(vector2, vector3) \
    BLEND_ROTATE_LEFT(vector3, vector4) \
    BLEND_ROTATE_LEFT(vector4, vector5) \
    ROTATE_LEFT(vector5)

#define BLEND_ROTATE6_LEFT(vector0, vector1, vector2, vector3, vector4, vector5, vector6) \
    BLEND_ROTATE_LEFT(vector0, vector1) \
    BLEND_ROTATE_LEFT(vector1, vector2) \
    BLEND_ROTATE_LEFT(vector2, vector3) \
    BLEND_ROTATE_LEFT(vector3, vector4) \
    BLEND_ROTATE_LEFT(vector4, vector5) \
    BLEND_ROTATE_LEFT(vector5, vector6) \
    ROTATE_LEFT(vector6)

#define BLEND_ROTATE7_LEFT(vector0, vector1, vector2, vector3, vector4, vector5, vector6, vector7) \
    BLEND_ROTATE_LEFT(vector0, vector1) \
    BLEND_ROTATE_LEFT(vector1, vector2) \
    BLEND_ROTATE_LEFT(vector2, vector3) \
    BLEND_ROTATE_LEFT(vector3, vector4) \
    BLEND_ROTATE_LEFT(vector4, vector5) \
    BLEND_ROTATE_LEFT(vector5, vector6) \
    BLEND_ROTATE_LEFT(vector6, vector7) \
    ROTATE_LEFT(vector7)

                      
//typedef __m128 vec4;
//
//typedef struct fv {  
//    union{  
//        float f[4];  
//        __m128 v;  
//    };  
//} f4v;


#if defined(__GNUC__)  && !defined(__INTEL_COMPILER)  

 
#elif defined __INTEL_COMPILER  

    inline void operator += (__m128& rhs, const __m128 B)
    {
     rhs = _mm_add_ps(rhs, B);
    }

    inline __m128 operator +(__m128& rhs, const __m128 B)
    {
     return _mm_add_ps(rhs, B);
    }    
    
    
#elif defined _MSC_VER  

#else  
    #error "nope"  
#endif  


#if defined(AMD)

inline __m128 _mm_dp_ps (__m128& __X, __m128& __Y, const int __M)
{
    //http://sseplus.sourceforge.net/group__emulated___s_s_e3.html
    
typedef __m128 float4;

extern float4 float4_mask_xy; // defined as { 0xFFFFFFFF, 0xFFFFFFFF, 0, 0 }
extern float4 float4_mask_xyz; // defined as { 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0 }
    
inline float4 dot2(float4 a, float4 b) {
    float4 temp = _mm_mul_ps(a, b);
    temp = _mm_and_ps(temp, float4_mask_xy);
    temp = _mm_hadd_ps(temp, temp);
    return _mm_hadd_ps(temp, temp);
}

inline float4 dot3(float4 a, float4 b) {
    float4 temp = _mm_mul_ps(a, b);
    temp = _mm_and_ps(temp, float4_mask_xyz);
    temp = _mm_hadd_ps(temp, temp);
    return _mm_hadd_ps(temp, temp);
}

inline float4 dot4(float4 a, float4 b) {
    float4 temp = _mm_mul_ps(a, b);
    temp = _mm_hadd_ps(temp, temp);
    return _mm_hadd_ps(temp, temp);
}
    
    
}

float ssedot(const float * __restrict u, const float *  __restrict v){  
    __m128 uv = _mm_mul_ps(_mm_load_ps(u), _mm_load_ps(v));  
    uv = _mm_hadd_ps(uv, uv); // or shuffle like there's no tomorrow   
    uv = _mm_hadd_ps(uv, uv); // if there ain't no haddps around.  
    return __builtin_ia32_vec_ext_v4sf(uv, 0);  
}  

#endif  




void opConvolve (const int s, const int w, const int h,
                 const int ks, const int kw, 
                 const float* input, float* output, const float* kernel); 
                 
void naiveConvolve (const int s, const int w, const int h, 
                    const int ks, const int kw, 
                    const float* input, float* output, const float* kernel); 

void alignedConvolve (const int s, const int w, const int h,
                      const int ks, const int kw, 
                      const float* input, float* output, const float* kernel); 
                      
void sseNoReuse1Convolve (const int s, const int w, const int h,
                      const int ks, const int kw, 
                      const float* input, float* output, const float* kernel); 

void sseNoReuse2Convolve (const int s, const int w, const int h,
                      const int ks, const int kw, 
                      const float* input, float* output, const float* kernel); 

void sseNoReuse3Convolve (const int s, const int w, const int h,
                      const int ks, const int kw, 
                      const float* input, float* output, const float* kernel); 

void sseNoReuse4Convolve (const int s, const int w, const int h, 
                          const int ks, int kw, 
                          const float* __restrict input, float* __restrict output, 
                          const float* __restrict kernel); 

//void sseNoReuse4Convolve2 (const int s, const int w, const int h,
//                           const int ks, const int kw, 
//                           const float* input, float* output, const float* kernel); 

void sseNoReuse5Convolve (const int s, const int w, const int h,
                      const int ks, const int kw, 
                      const float* input, float* output, const float* kernel); 

void sseNoReuse6Convolve (const int s, const int w, const int h,
                      const int ks, const int kw, 
                      const float* input, float* output, const float* kernel); 

void sseNoReuse7Convolve (const int s, const int w, const int h,
                      const int ks, const int kw, 
                      const float* input, float* output, const float* kernel); 

void sseReuse1Convolve (const int s, const int w, const int h,
                      const int ks, const int kw, 
                      const float* input, float* output, const float* kernel); 

void sseReuse2Convolve (const int s, const int w, const int h,
                      const int ks, const int kw, 
                      const float* input, float* output, const float* kernel); 

void sseReuse3Convolve (const int s, const int w, const int h,
                      const int ks, const int kw, 
                      const float* input, float* output, const float* kernel); 

void sseReuse4Convolve (const int s, const int w, const int h,
                      const int ks, const int kw, 
                      const float* input, float* output, const float* kernel); 

void sseReuse5Convolve (const int s, const int w, const int h,
                      const int ks, const int kw, 
                      const float* input, float* output, const float* kernel); 

void sseReuse6Convolve (const int s, const int w, const int h,
                      const int ks, const int kw, 
                      const float* input, float* output, const float* kernel); 

void sseReuse7Convolve (const int s, const int w, const int h,
                      const int ks, const int kw, 
                      const float* input, float* output, const float* kernel);                       

void unalignedSSEConvolve (const int s, const int w, const int h,
                      const int ks, const int kw, 
                      const float* input, float* output, const float* kernel); 

void pointerArithmeticConvolve (const int s, const int w, const int h, 
                                const int ks, const int kw, 
                                const float* input, float* output, const float* kernel);

void loopUnrollConvolve (const int s, const int w, const int h, 
                         const int ks, const int kw, 
                         const float* __restrict input, float* __restrict output, const float* kernel);

void prefetchConvolve64 (const int s, const int w, const int h, 
                         const int ks, const int kw, 
                         const float* input, float* output, const float* kernel); 
                         
void prefetchConvolve128 (const int s, const int w, const int h, 
                          const int ks, const int kw, 
                          const float* input, float* output, const float* kernel);                                                    
                          
float convolution(const float* __restrict input, const int s, 
                         const float* __restrict kernel, const int ks, const int kw, 
                         const int x, const int y);

                   
void loopBlockConvolve (const int s, const int w, const int h, 
                        const int ks, const int kw, 
                        const float* input, float* output, const float* kernel, 
                        const int xBlock, const int yBlock);
                        
void loopBlockLoopUnrollConvolve (const int s, const int w, const int h, 
                        const int ks, const int kw, 
                        const float* input, float* output, const float* kernel, 
                        const int xBlock, const int yBlock);                        
                                                               
void loopBlockAlignedSSEConvolve (const int s, const int w, const int h, 
                        const int ks, const int kw, 
                        const float* input, float* output, const float* kernel, 
                        const int xBlock, const int yBlock);
                                                               
void loopBlockAlignedSSEConvolve2 (const int s, const int w, const int h, 
                        const int ks, const int kw, 
                        const float* input, float* output, const float* kernel, 
                        const int xBlock, const int yBlock);
                                                               

                             
                                       
void sse3Convolve (const int s, const int w, const int h, const int ks, 
                   const float* input, float* output, const float* kernel);
                   
void sse5Convolve (const int s, const int w, const int h, const int ks, 
                   const float* input, float* output, const float* kernel);
                   
void sse7Convolve (const int s, const int w, const int h, const int ks, 
                   const float* input, float* output, const float* kernel);
                   
void sse9Convolve (const int s, const int w, const int h, const int ks, 
                   const float* input, float* output, const float* kernel);
                   
void sse11Convolve (const int s, const int w, const int h, const int ks, 
                   const float* input, float* output, const float* kernel);
                   
void sseWideKernelConvolve (const int s, const int w, const int h, 
                            const int ks, const int kw, 
                            const float* input, float* output, const float* kernel); 
                            
void sse3CmConvolve (const int s, const int w, const int h, const int ks, 
                   const float* input, float* output, const float* kernel);
                   
void sse3LbConvolve (const int s, const int w, const int h, const int ks, 
                   const float* input, float* output, const float* kernel);
                   
void separableConvolve (const int s, const int w, const int h, const int kw, 
                        const float* __restrict input, float* __restrict output, 
                        const float* __restrict kernelX, 
                        const float* __restrict kernelY); 
                        
//void separableConvolve2 (const int s, const int w, const int h, const int kw, 
//                        const float* input, float* output, const float* kernelX, const float* kernelY);
                        
void separableLoopBlockConvolve (const int s, const int w, const int h, const int kw, 
                        const float* input, float* output, const float* kernelX, const float* kernelY, 
                        const int xBlock, const int yBlock);

void scSSE (const int s, const int w, const int h, int kw, 
            const float* input, float* output, 
            const float* kernelX, const float* kernelY);

                                                                                       
void sc3SSE (const int s, const int w, const int h, 
             const float* input, float* output, 
             const float* kernelX, const float* kernelY);
                                       
void sc5SSE (const int s, const int w, const int h, 
             const float* input, float* output, const float* kernelX, const float* kernelY);
                                       
void sc7SSE (const int s, const int w, const int h, 
             const float* input, float* output, const float* kernelX, const float* kernelY);
                                       
void sc9SSE (const int s, const int w, const int h, 
             const float* input, float* output, const float* kernelX, const float* kernelY);


                                  
void scGaussian5SSE (const int s, const int w, const int h, 
                     const float* input, float* output, 
                     const float* kernel);
                     
void scGaussian7SSE (const int s, const int w, const int h, 
                     const float* input, float* output, 
                     const float* kernel);
                     
void scGaussian9SSE (const int s, const int w, const int h, 
                     const float* input, float* output, 
                     const float* kernel);       
                     
void processBoundaries2D(const int s, const int w, const int h, 
                       const int ks, int kw, 
                       const float* input, float* output, const float* kernel);

void processBoundariesS2D(const int s, const int w, const int h, 
                        const int kw, 
                        const float* input, float* output, const float* kernelX, const float* kernelY);
                     

//helper functions

void printImage(int w, int h, int s, const float* out);
void printImageToFile(const std::string& file, int width, int height, int s, const float* out);
void printKernel2D(const int ks, const int kw, const float* kernel);
void clear2DBuffer( float* buffer, int s, int height );
float* allocateFloatAlignedBuffer (int width, int height);
int calculateAlignedStride (int width, int pixelSizeInBytes, int alignInBytes);



#endif // _OPCONVUTIONFILTER_H