/*!
 * @file 		IOThread.cpp
 * @author 		Zdenek Travnicek
 * @date 		30.10.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef TYPES_H_
#define TYPES_H_
#ifndef YURI_USE_CXX11
#error C++11 mode is required now.
#endif
#include "yuri/core/platform.h"
#include "yuri/core/uvector.h"
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <chrono>
#include <cassert>
#include "utils.h"

//! All yuri related stuff belongs here
namespace yuri {
typedef uint64_t usize_t;
typedef int64_t ssize_t;
typedef uint32_t uint_t;
typedef int32_t sint_t;
typedef uint16_t ushort_t;
typedef int16_t sshort_t;
typedef uint8_t ubyte_t;
typedef int8_t sbyte_t;

using std::shared_ptr;
using std::weak_ptr;
using std::make_shared;
using std::mutex;
using std::unique_ptr;

typedef std::unique_lock<std::mutex> lock_t;
#ifndef YURI_ANDROID
using std::timed_mutex;
typedef std::unique_lock<std::timed_mutex> timed_lock;
#else
typedef std::mutex timed_mutex;
typedef lock_t timed_lock;
#endif
using std::dynamic_pointer_cast;
using std::static_pointer_cast;
using std::thread;
using std::enable_shared_from_this;
typedef std::chrono::nanoseconds time_duration;
const std::chrono::nanoseconds time_duration_infinity (std::chrono::duration_values<time_duration::rep>::max());
typedef std::chrono::time_point<std::chrono::steady_clock, time_duration> time_value;
using std::chrono::nanoseconds;
using std::chrono::microseconds;
namespace this_thread {
template<class Rep, class Period>
void sleep(const std::chrono::duration<Rep, Period>& dur) { std::this_thread::sleep_for(dur);}
}
using std::this_thread::get_id;

typedef usize_t size_t;
typedef ssize_t format_t;
typedef size_t threadId_t;

typedef yuri::uvector<yuri::ubyte_t,false> plane_t;


}
#ifdef YURI_LINUX
#define YURI_UNUSED __attribute__((unused))
#else
#define YURI_UNUSED
#endif
#endif /* TYPES_H_ */
