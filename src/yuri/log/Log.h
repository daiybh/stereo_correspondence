/*!
 * @file 		Log.h
 * @author 		Zdenek Travnicek
 * @date 		24.7.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2010 - 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef LOG_H_
#define LOG_H_
#include "yuri/core/utils/new_types.h"
#include "LogProxy.h"

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <atomic>

//! @namespace yuri All yuri related stuff belongs here
namespace yuri
{
//! @namespace yuri::log All logging related stuff belongs here
namespace log {

/**
 * @brief Flags representing debug levels and flags
 */
enum debug_flags_t: long {
	silent 			=	0,
	fatal 			=	1,                                                    //!< Fatal error, application will probably quit
	error	 		=	2,                                                    //!< Fatal error, application will probably quit
	warning	 		=	3,                                                  //!< Warning, application should continue (but may not work correctly)
	info			=	4,                                                      //!< Information about execution
	debug 			=	5,                                                    //!< Debug information not needed during normal usage
	verbose_debug	=	6,                                               //!< Verbose debug, used for debugging only
	trace			=	7,                                                     //!< Tracing information, should never be needed during normal work
	flag_mask		=   7,//!< Mask representing all flags
	show_time		=	1L << 4,                                                 //!< Enables outputing actual time with the output
	show_thread_id	=	1L << 5,                                             //!< Enables showing thread id
	show_level		=	1L << 6,                                                //!< Enables showing debug level name
	use_colors		=	1L << 7                                                 //!< Enable usage of colors
};

typedef debug_flags_t  debug_flags;

/**
 * @brief Main logging facility for libyuri
 */
class Log
{
public:
	//! Constructs Log instance with @em out as a backend
	EXPORT Log(std::ostream &out);
	//! Constructs Log instance as a copy of @em log, with a new id
	EXPORT Log(const Log& log);

	Log& operator=(Log&& rhs) noexcept;
	EXPORT virtual ~Log() noexcept;
//	EXPORT void set_id(int id);
	EXPORT void set_label(std::string s);
	EXPORT void set_flags(long f) { output_flags_=f; }
	EXPORT LogProxy<char> operator[](debug_flags f) const;
	EXPORT long get_flags() { return output_flags_; }
	EXPORT void set_quiet(bool q) {quiet_ =q;}
	EXPORT void adjust_log_level(long delta);
private:
	// Global counter for IDs
	static std::atomic<int> uids;
	// ID of current Log
	int uid;
	// Output stream
	yuri::shared_ptr<guarded_stream<char> > out;
	yuri::mutex lock;
	std::string logger_name_;
	long output_flags_;
	bool quiet_;
	std::string print_time() const;
	std::string print_level(debug_flags flags) const ;
};

}
}
#endif /*LOG_H_*/
