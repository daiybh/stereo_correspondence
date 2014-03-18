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

//! @namespace yuri All yuri related stuff belongs here
namespace yuri
{
//! @namespace yuri::log All logging related stuff belongs here
namespace log {

/**
 * @brief Flags representing debug levels and flags
 */
enum _debug_flags {
	fatal 			=	1L << 0,                                                    //!< Fatal error, application will probably quit
	error	 		=	1L << 1,                                                    //!< Fatal error, application will probably quit
	warning	 		=	1L << 2,                                                  //!< Warning, application should continue (but may not work correctly)
	info			=	1L << 3,                                                      //!< Information about execution
	debug 			=	1L << 4,                                                    //!< Debug information not needed during normal usage
	verbose_debug	=	1L << 5,                                               //!< Verbose debug, used for debugging only
	trace			=	1L << 6,                                                     //!< Tracing information, should never be needed during normal work
	flag_mask		= fatal+error+warning+info+debug+verbose_debug+trace,//!< Mask representing all flags
	show_time		=	1L << 16,                                                 //!< Enables outputing actual time with the output
	show_thread_id	=	1L << 17,                                             //!< Enables showing thread id
	show_level		=	1L << 18,                                                //!< Enables showing debug level name
	use_colors		=	1L << 19                                                 //!< Enable usage of colors
};

typedef _debug_flags  debug_flags;

/**
 * @brief Main logging facility for libyuri
 */
class EXPORT Log
{
public:
	//! Constructs Log instance with @em out as a backend
	Log(std::ostream &out);
	//! Constructs Log instance as a copy of @em log, with a new id
	Log(const Log& log);
	virtual ~Log() noexcept;
	void set_id(int id);
	void set_label(std::string s);
	void set_flags(int f) { output_flags=f; }
	LogProxy<char> operator[](debug_flags f);
	int get_flags() { return output_flags; }
	void set_quiet(bool q) {quiet =q;}
private:
	static int uids;
	int uid;
	yuri::shared_ptr<guarded_stream<char> > out;
	int id;
	yuri::mutex lock;
	std::string ids;
	debug_flags flags;
	int output_flags;
	bool quiet;
	void set_flag(debug_flags f);
	std::string print_time();
	std::string print_level();

};

}
}
#endif /*LOG_H_*/
