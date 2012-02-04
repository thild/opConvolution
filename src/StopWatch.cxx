// 
// StopWatch.cxx
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

#include "StopWatch.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

#include <ctime>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>


StopWatch::StopWatch() : _running(false), _cycles(0), _start(0)
{

#ifdef _WIN32

    QueryPerformanceFrequency((LARGE_INTEGER *)&_frequency);
 
#else
    _frequency = 1000000;
#endif

}

StopWatch::~StopWatch()
{
    // EMPTY!
}

void
StopWatch::Start(void)
{
    // Return immediately if the timer is already _running
    if (_running) return;
    _running = true;
    _start = Now();

}

void
StopWatch::StartNew(void)
{
    Reset();
    Start();
}

i64 StopWatch::Now(void) {
    i64 n;
    #ifdef _WIN32
        QueryPerformanceCounter((LARGE_INTEGER *)&n);
    #else
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        n = (i64)ts.tv_sec * 1000000LL + (i64)ts.tv_nsec / 1000LL;
    
//        struct timeval s;
//        gettimeofday(&s, 0);
//        n = (i64)s.tv_sec * 1000 + (i64)s.tv_usec / 1000;
    #endif
    return n;
}

void
StopWatch::Stop(void)
{
    if(!_running) return;
    _running = false;
    i64 n = Now();
    n -= _start;
    _cycles += n;
    _start = 0;
}

void
StopWatch::AddCheckpoint(const string tag)
{
    AddCheckpoint(tag, false);
}

void
StopWatch::AddCheckpoint(const string tag, const bool ignore)
{
    if(ignore) {
        AddCheckpoint(tag, ignore, -1);
    }
    else {
        AddCheckpoint(tag, ignore, LastNotIgnoredCheckpointIndex());
    }
}

unsigned int StopWatch::LastNotIgnoredCheckpointIndex() {
    return  GetNotIgnoredCheckpoints().size();
}

void
StopWatch::AddCheckpoint(const string tag, const bool ignore, const int position)
{
    double instant = GetElapsedTime();
    double elapsed = 0;
    if(_checkpoints.size() > 0)
        elapsed = instant - _checkpoints[_checkpoints.size() - 1].Instant;
    _checkpoints.push_back(Checkpoint(tag, instant, elapsed, ignore, position));
}

//returns only not ignored
double StopWatch::GetTotalCheckpointsTime() {
    double total = 0;
    for (vector<Checkpoint>::iterator it = _checkpoints.begin(); it != _checkpoints.end(); it++ ) { 
        if(!it->Ignore) {
            total += it->Elapsed;
        }
    }
    return total;
}

//returns only not ignored
vector<Checkpoint> StopWatch::GetNotIgnoredCheckpoints() {
    vector<Checkpoint> v;
    for (vector<Checkpoint>::iterator it = _checkpoints.begin(); it != _checkpoints.end(); it++ ) { 
        if(!it->Ignore) {
            v.push_back(*it);
        }
    }
    return v;
}


void
StopWatch::Reset(void)
{
    _running = false;
    _cycles = 0;
    _start = 0;
    _checkpoints.clear();
}

double
StopWatch::GetElapsedTime(void)
{
    if (_running) {
        i64 n = Now();
        n -= _start;
        return (double)(_cycles + n) / (double)_frequency;
    }
    return (double)_cycles / (double)_frequency;
}



inline std::string StopWatch::ToString()
{
  std::ostringstream os;
  os << "Elapsed time [" << std::setiosflags(std::ios::fixed)
            << std::setprecision(2)
            << this->GetElapsedTime() << "] seconds\n";
  return  os.str();

}

inline std::ostream& operator<<(std::ostream& os, StopWatch& t)
{
  os << std::setprecision(4) << std::setiosflags(std::ios::fixed)
    << t.GetElapsedTime();
  return os;
}


vector<Checkpoint>& StopWatch::GetCheckpoints()
{
  return _checkpoints;
}

inline std::ostream& operator<<(std::ostream& os, Checkpoint& t)
{
  os <<  std::setprecision(4) << std::setiosflags(std::ios::fixed)
    << "Tag: " << t.Tag << "; Instant: " << t.Instant  << "; Elapsed: " << t.Elapsed;
  return os;
}

StopWatch* StopWatchPool::GetStopWatch(const string tag) {
    static map<string, StopWatch> instances;
    return &instances[tag];
}
