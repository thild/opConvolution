// 
// StopWatch.h
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

#ifndef _STOPWATCH_H_
#define _STOPWATCH_H_
/**
 * \file StopWatch.h
 * \brief Class for measuring time.
 */
#ifdef _WIN32
/**
 * \typedef __int64 i64
 * \brief Maps the windows 64 bit integer to a uniform name
 */
typedef __int64 i64 ;
#else
/**
 * \typedef long long i64
 * \brief Maps the linux 64 bit integer to a uniform name
 */
typedef long long i64;
#endif

#include <iostream>
#include <vector>
#include <map>

using namespace std;

class Checkpoint 
{
  public:
    string Tag;
    double Instant;
    double Elapsed;
    bool Ignore;
    Checkpoint(string tag, double instant, double elapsed, bool ignore) {
      this->Tag = tag;
      this->Instant = instant;
      this->Elapsed = elapsed;
      this->Ignore = ignore;
    }
};


/**
 * \class StopWatch
 * \brief Counter that provides a fairly accurate timing mechanism for both
 * windows and linux. This timer is used extensively in all the samples.
 */
class StopWatch {

public:
    /**
     * \fn StopWatch()
     * \brief Constructor for StopWatch that initializes the class
     */
    StopWatch();
    /**
     * \fn ~StopWatch()
     * \brief Destructor for StopWatch that cleans up the class
     */
    ~StopWatch();
    
    /**
     * \fn void Start(void)
     * \brief Start the timer
     * \sa Stop(), Reset()
     */
    void Start(void);
    void StartNew(void);
    /**
     * \fn void Stop(void)
     * \brief Stop the timer
     * \sa Start(), Reset()
     */
    void Stop(void);
    /**
     * \fn void AddCheckpoint(const string tag)
     * \brief Add a checkpoint
     */
    void AddCheckpoint(const string tag);
    void AddCheckpoint(const string tag, const bool ignore);
    
    double GetTotalCheckpointsTime();
    
    /**
     * \fn void Reset(void)
     * \brief Reset the timer to 0
     * \sa Start(), Stop()
     */
    void Reset(void);
    /**
     * \fn double GetElapsedTime(void)
     * \return Amount of time that has accumulated between the \a Start()
     * and \a Stop() function calls
     */
    double GetElapsedTime(void);
    
    vector<Checkpoint>& GetCheckpoints();
    vector<Checkpoint> GetNotIgnoredCheckpoints();
    
    std::string ToString();
    
  private:

    /**
     * \fn i64 Now(void)
     * \brief Return now time
     */
    i64 Now(void);
    
    bool _running;
    i64 _frequency;
    i64 _cycles;
    i64 _start;
    vector<Checkpoint> _checkpoints;
};


class StopWatchPool {
public:
    static StopWatch* GetStopWatch(const string tag);
    StopWatchPool();
    ~StopWatchPool();
  private:
};

template <class T>
class Singleton
{
public:
  static T* Instance() {
      if(!m_pInstance) m_pInstance = new T;
      assert(m_pInstance !=NULL);
      return m_pInstance;
  }
protected:
  Singleton();
  ~Singleton();
private:
  Singleton(Singleton const&);
  Singleton& operator=(Singleton const&);
  static T* m_pInstance;
};

template <class T> T* Singleton<T>::m_pInstance=NULL;

#endif // _STOPWATCH_H_
