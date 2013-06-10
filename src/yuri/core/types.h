/*!
 * @file 		BasicIOThread.cpp
 * @author 		Zdenek Travnicek
 * @date 		30.10.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef TYPES_H_
#define TYPES_H_
#include "yuri/core/platform.h"
#include "yuri/core/uvector.h"
#ifndef YURI_USE_CXX11
#include <boost/cstdint.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/smart_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/thread/thread.hpp>
#include <boost/assign.hpp>
#include <boost/enable_shared_from_this.hpp>
#include "boost/lexical_cast.hpp"
#include <boost/posix_time.hpp>
#else
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <chrono>
#include <cassert>
#include "utils.h"
#endif
//! All yuri related stuff belongs here
namespace yuri {
#ifndef YURI_USE_CXX11
typedef boost::uint64_t usize_t;
typedef boost::int64_t ssize_t;
typedef boost::uint32_t uint_t;
typedef boost::int32_t sint_t;
typedef boost::uint16_t ushort_t;
typedef boost::int16_t sshort_t;
typedef boost::uint8_t ubyte_t;
typedef boost::int8_t sbyte_t;

using boost::shared_ptr;
using boost::weak_ptr;
using boost::make_shared;
using boost::mutex;
using boost::timed_mutex;
using boost::dynamic_pointer_cast;
using boost::thread;
typedef boost::mutex::scoped_lock lock;
using boost::assign::map_list_of;
using boost::assign::list_of;
using boost::enable_shared_from_this;
using boost::lexical_cast;
using boost::bad_lexical_cast;
using boost::iequals;
using boost::posix_time::time_duration;
const time_duration time_duration_infinity = boost::posix_time::pos_infin;
typedef boost::posix_time::ptime timeval;
using boost::posix_time::microseconds;
using boost::posix_time::nanoseconds;
namespace this_thread {
using boost::this_thread::sleep;
}
#else
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
typedef std::unique_lock<std::mutex> lock;
#ifndef YURI_ANDROID
using std::timed_mutex;
typedef std::unique_lock<std::timed_mutex> timed_lock;
#else
typedef std::mutex timed_mutex;
typedef lock timed_lock;
#endif
using std::dynamic_pointer_cast;
using std::thread;
using std::enable_shared_from_this;
typedef std::chrono::nanoseconds time_duration;
const std::chrono::nanoseconds time_duration_infinity (std::chrono::duration_values<time_duration::rep>::max());
typedef std::chrono::time_point<std::chrono::steady_clock> time_value;
using std::chrono::nanoseconds;
using std::chrono::microseconds;
namespace this_thread {
template<class Rep, class Period>
void sleep(const std::chrono::duration<Rep, Period>& dur) { std::this_thread::sleep_for(dur);}
}

#endif
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
