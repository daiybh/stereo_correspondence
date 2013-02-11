#ifndef LOG_H_
#define LOG_H_
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <boost/thread.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/shared_ptr.hpp>
#include "LogProxy.h"
#include "yuri/io/types.h"
namespace yuri
{
namespace log {
using namespace std;
using boost::shared_ptr;

enum _debug_flags {
	fatal 			=	1L << 0,
	error	 		=	1L << 1,
	warning	 		=	1L << 2,
	info			=	1L << 3,
	normal	 		=	1L << 4,
	debug 			=	1L << 5,
	verbose_debug	=	1L << 6,
	trace			=	1L << 7,
	flag_mask		= fatal+error+warning+info+normal+debug+verbose_debug+trace,
	show_time		=	1L << 16,
	show_thread_id	=	1L << 17,
	show_level		=	1L << 18,
	use_colors		=	1L << 19
};

typedef _debug_flags  debug_flags;

class Log
{
public:
	EXPORT Log(std::ostream &out);
	EXPORT Log(const Log& log);
	EXPORT virtual ~Log();
	EXPORT void setID(int id);
//	EXPORT template <class T> std::ostream& operator<<(T &t);
	EXPORT void setLabel(std::string s);
	EXPORT void setFlags(int f) { output_flags=f; }
	EXPORT LogProxy operator[](debug_flags f);
	EXPORT int get_flags() { return output_flags; }
	EXPORT void set_quiet(bool q) {quiet =q;}
protected:
	static int uids;
	int uid;
	shared_ptr<guarded_stream> out;
	int id;
	boost::mutex lock;
	//char *ids;
	string ids;
	debug_flags flags;
	int default_flags, output_flags;
	bool quiet;
	void set_flag(debug_flags f);
	EXPORT string print_time();
	EXPORT string print_level();

};


//template <class T> std::ostream &Log::operator <<(T &t)
//{
//	boost::mutex::scoped_lock l(lock);
//	if (flags > (output_flags & flag_mask)) {
//		if (null) return *null;
//		else return *this;
//	}
//	if (!quiet) {
//		return out << ((output_flags&show_level)?print_level():"") << id << ":" << uid << ": "  << ids << print_time() << t;
//	}
//	else return out << t;
//}

}
}
#endif /*LOG_H_*/
