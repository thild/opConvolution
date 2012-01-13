// 
// timer.h
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

#ifndef _TIMER_H
#define _TIMER_H

#ifdef _WIN32
#include <windows.h>
typedef __int64 int64 ;
#else
#include <sys/time.h>
typedef long long int64;
#endif

#include <ctime>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>


class timer
{
 friend std::ostream& operator<<(std::ostream& os, timer& t);

 private:
  bool _running;
  int64 _frequency;
  int64 _cycles;
  int64 _start;
  void refresh();

 public:

  timer();
  ~timer();

  void start();
  void reset();
  void stop();
  std::string toString();
  double elapsed();
}; // class timer


  // '_running' is initially false.  A timer needs to be explicitly started
  // using 'start' or 'reset'
  timer::timer() : _running(false), _cycles(0), _start(0)   
  {
#ifdef _WIN32
    QueryPerformanceFrequency((LARGE_INTEGER *)&_frequency);
#else
    _frequency = 1000;
#endif
  }

  timer::~timer()
  {
	  
  }

//===========================================================================
// Return the total time that the timer has been in the "_running"
// state since it was first "started" or last "reseted".  For
// "short" time periods (less than an hour), the actual cpu time
// used is reported instead of the elapsed time.

inline double timer::elapsed()
{
  if (_running) refresh();
  return (double)_cycles / (double)_frequency;
} // timer::elapsed


//===========================================================================
// Start a timer.  If it is already _running, let it continue _running.
// Print an optional message.

inline void timer::start()
{
  // Return immediately if the timer is already _running
  if (_running) return;

  _running = true;

#ifdef _WIN32

    QueryPerformanceCounter((LARGE_INTEGER *)&_start);

#else

    struct timeval s;
    gettimeofday(&s, 0);
    _start = (int64)s.tv_sec * 1000 + (int64)s.tv_usec / 1000;

#endif

} // timer::start


//===========================================================================
// Turn the timer off and start it again from 0.  Print an optional message.

inline void timer::reset()
{
  // Set timer status to _running, reset accumulated time, and set start time
  _running = true;
  _cycles = 0;
} // timer::reset

//===========================================================================
// Stop the timer and print an optional message.

inline void timer::stop()
{
  // Compute accumulated _running time and set timer status to not _running
  _running = false;
  refresh();
} // timer::stop


inline void timer::refresh()
{
  int64 n;

#ifdef _WIN32

    QueryPerformanceCounter((LARGE_INTEGER *)&n);

#else

    struct timeval s;
    gettimeofday(&s, 0);
    n = (int64)s.tv_sec * 1000 + (int64)s.tv_usec / 1000;

#endif

    n -= _start;
    _cycles = n;

} // timer::refresh


//===========================================================================
// Print out an optional message followed by the current timer timing.

inline std::string timer::toString()
{
  std::ostringstream os;
  os << "Elapsed time [" << std::setiosflags(std::ios::fixed)
            << std::setprecision(2)
            << elapsed() << "] seconds\n";
  return  os.str();

} // timer::elapsed

//===========================================================================
// Allow timers to be printed to ostreams using the syntax 'os << t'
// for an ostream 'os' and a timer 't'.  For example, "cout << t" will
// print out the total amount of time 't' has been "_running".

inline std::ostream& operator<<(std::ostream& os, timer& t)
{
  os << std::setprecision(4) << std::setiosflags(std::ios::fixed)
    << t.elapsed();
  return os;
}

//===========================================================================

#endif // TIMER_H

