/*!
 * @file 		Log.cpp
 * @author 		Zdenek Travnicek
 * @date 		24.7.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "Log.h"
#include <map>
#include <boost/assign.hpp>
namespace yuri
{
namespace log
{


int Log::uids=0;

namespace {
std::map<_debug_flags, std::string> level_names=boost::assign::map_list_of<_debug_flags, std::string>
	(fatal,"FATAL ERROR")
	(error,"ERROR")
	(warning,"WARNING")
	(info,"INFO")
	(normal,"")
	(debug,"DEBUG")
	(verbose_debug,"VERBOSE_DEBUG")
	(trace,"TRACE");

std::map<_debug_flags, std::string> level_colors=boost::assign::map_list_of<_debug_flags, std::string>
(fatal,"\033[4;31;42m") // Red, underscore, bg
(error,"\033[31m") // Red
(warning,"\033[35m")
(info,"\033[32m")
(normal,"\033[00m")
(debug,"\033[35m")
(verbose_debug,"\033[36m")
(trace,"\033[4;36m");
}
Log::Log(std::ostream &out):uid(uids++),out(new guarded_stream(out)),
#ifdef USE_MPI_
id(-1),
#else
id(0),
#endif
ids(""),flags(normal),output_flags(normal),quiet(false)
{
}

Log::~Log()
{
	(*this)[verbose_debug] << "Destroying logger " << uid << std::endl;
}

void Log::setID(int id)
{
	this->id=id;
}

Log::Log(const Log &log):uid(uids++),out(log.out),id(log.id),ids(""),flags(log.flags),
		output_flags(log.output_flags),quiet(log.quiet)
{
	(*this)[verbose_debug] << "Copying logger "	<< log.uid << " -> " << uid	<< std::endl;
}

void Log::setLabel(std::string s)
{
	ids=s;
}

LogProxy Log::operator[](debug_flags f)
{
	set_flag(f);
	LogProxy lp(*out,flags > (output_flags & flag_mask));
	if (!quiet) {
		lp << ((output_flags&show_level)?print_level():"") << id << ":" << uid << ": "  << ids << print_time();
	}
	return lp;
}

void Log::set_flag(debug_flags f)
{
	flags=f;
}


std::string Log::print_time()
{
	std::string s;
	std::stringstream ss;
	if (output_flags & show_thread_id) {
		 ss << boost::this_thread::get_id();
		 s += std::string(" ") + ss.str();
	}
	if (output_flags & show_time) s += std::string(" ") + boost::posix_time::to_simple_string(boost::posix_time::microsec_clock::local_time().time_of_day());
	if (s.length()) s+=" ";
	return s;
}

std::string Log::print_level()
{
	_debug_flags f = static_cast<_debug_flags>(flags&flag_mask);
	if (level_names.count(f)) {
		if ((output_flags&use_colors) && level_colors.count(f)) {
			return level_colors[f]+level_names[f]+level_colors[normal]+" ";
		}
		return level_names[f]+" ";
	}
	return std::string();
}
}
}
