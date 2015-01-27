/*!
 * @file 		Log.cpp
 * @author 		Zdenek Travnicek
 * @date 		24.7.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2010 - 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "Log.h"
#include <map>
#include "yuri/core/utils.h"
#if !defined YURI_ANDROID && !defined(YURI_WIN)
#include <boost/date_time/posix_time/posix_time.hpp>
#endif
namespace yuri
{
namespace log
{


int Log::uids=0;

namespace {
const std::map<_debug_flags, std::string> level_names= {
	{fatal,"FATAL ERROR"},
	{error,"ERROR"},
	{warning,"WARNING"},
	{info,"INFO"},
	{debug,"DEBUG"},
	{verbose_debug,"VERBOSE_DEBUG"},
	{trace,"TRACE"}};

const std::map<_debug_flags, std::string> level_colors = {
std::map<_debug_flags, std::string> level_colors = {
#ifdef YURI_LINUX
{fatal,"\033[4;31;42m"}, // Red, underscore, bg
{error,"\033[31m"}, // Red
{warning,"\033[35m"},
{info,"\033[00m"},
{debug,"\033[35m"},
{verbose_debug,"\033[36m"},
{trace,"\033[4;36m"}
#endif
};
}

/**
 * Creates a new Log instance
 * @param out Stream to output to
 */
Log::Log(std::ostream &out):uid(uids++),out(new guarded_stream<char>(out)),
#ifdef USE_MPI_
id(-1),
#else
id(0),
#endif
ids(""),output_flags(info),quiet(false)
{
}

/**
 * Destroys an instance
 */
Log::~Log() noexcept
{
	(*this)[verbose_debug] << "Destroying logger " << uid;
}

void Log::set_id(int new_id)
{
	id=new_id;
}

/**
 * Creates a new Log instance as a copy of @em log.  The instance will get an unique uid.
 * @param log Log instance to copy from
 */
Log::Log(const Log &log):uid(uids++),out(log.out),id(log.id),ids(""),
		output_flags(log.output_flags),quiet(log.quiet)
{
	(*this)[verbose_debug] << "Copying logger "	<< log.uid << " -> " << uid;
}
/**
 * Sets textual label for current instance
 * @param s new label
 */
void Log::set_label(std::string s)
{
	ids=s;
}

/**
 * Return an instance of LogProxy that should be used for current level. It should not outlive the original Log object!
 * @param f flag to compare with. If this flag represents log level higher than current, the LogProxy object will be 'dummy'
 * @return An instance of LogProxy
 */
LogProxy<char> Log::operator[](debug_flags f)
{
//	set_flag(f);
	LogProxy<char> lp(*out,f /*flags*/ > (output_flags & flag_mask));
	if (!quiet) {
		lp << ((output_flags&show_level)?print_level(f):"") << id << ":" << uid << ": "  << ids << print_time();
	} else {
		lp << " ";
	}
	return lp;
}

//void Log::set_flag(debug_flags f)
//{
//	flags=f;
//}


std::string Log::print_time() const
{
	std::string s;
	std::stringstream ss;
#if !defined(YURI_ANDROID) && !defined(YURI_WIN)
	if (output_flags & show_thread_id) {
//		 ss << yuri::get_id();
		 s += std::string(" ") + ss.str();
	}
	if (output_flags & show_time) s += std::string(" ") + boost::posix_time::to_simple_string(boost::posix_time::microsec_clock::local_time().time_of_day());
#endif
	if (s.length()) s+=" ";
	return s;
}

std::string Log::print_level(debug_flags flags) const
{
	debug_flags f = static_cast<debug_flags>(flags&flag_mask);
	auto it = level_names.find(f);
//	if (level_names.count(f)) {
	if (it == level_names.end()) {
		return {};
	}
	const auto& name = it->second;

	if (output_flags&use_colors) {
		auto it_col = level_colors.find(f);
		auto it_col2 = level_colors.find(info);
		if (it_col!=level_colors.end() && it_col2 != level_colors.end()) {
			const auto& color1 = it_col->second;
			const auto& color2 = it_col2->second;
			return color1 +  name + color2 +" ";
		}
	}
	return name+" ";
}
}
}
