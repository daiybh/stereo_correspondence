/*
 * BasicPipe.h
 *
 *  Created on: Jul 28, 2010
 *      Author: neneko
 */

#ifndef BASICPIPE_H_
#define BASICPIPE_H_
#include <queue>
#include <string>
#include <map>
#include <cassert>
#include <sys/types.h>
#ifdef __linux__
#include <sys/socket.h>
#else
#ifdef __WIN32__
#include <winsock2.h>
#include <windows.h>
#else
#error "Unsupported platform"
#endif
#endif
#include <boost/foreach.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include "yuri/log/Log.h"
#include "BasicFrame.h"
#include "pipe_types.h"
#include "yuri/config/Parameters.h"
#include "yuri/threads/ThreadBase.h"

namespace yuri {

namespace io {
using namespace yuri::log;
using namespace yuri::config;
using namespace yuri::threads;



struct compare_insensitive {
	bool operator() (const std::string a, const std::string b) const
			{ return boost::ilexicographical_compare(a,b); }
};

class BasicPipe {
public:
	BasicPipe(Log &log_,std::string name);
	virtual ~BasicPipe();

	//virtual void push_frame(BasicFrame *frame);
	virtual void push_frame(pBasicFrame frame);

	virtual pBasicFrame pop_frame();
	virtual pBasicFrame pop_latest();

	virtual  void set_type(yuri::format_t type);
	virtual  yuri::format_t get_type();


	virtual yuri::size_t get_size() { return bytes; }
	virtual yuri::size_t get_count() { return count; }

	virtual bool is_changed();
	virtual bool is_empty();

	virtual void close();
	virtual bool is_closed();

	virtual void set_limit(yuri::size_t limit0, yuri::ubyte_t policy=YURI_DROP_SIZE);

	virtual int get_notification_fd();
	virtual void cancel_notifications();

public:
	EXPORT static shared_ptr<BasicPipe> generator(Log &log, std::string name, Parameters &parameters);
	EXPORT static shared_ptr<Parameters> configure();
	EXPORT static std::map<yuri::format_t, const FormatInfo_t> formats;
	EXPORT static boost::mutex format_lock;
	EXPORT static std::string get_type_string(yuri::format_t type);
	EXPORT static std::string get_format_string(yuri::format_t type);
	EXPORT static std::string get_simple_format_string(yuri::format_t type);
	EXPORT static yuri::size_t get_bpp_from_format(yuri::format_t type);
	EXPORT static FormatInfo_t get_format_info(yuri::format_t format);
	EXPORT static yuri::format_t get_format_group(yuri::format_t format);
	EXPORT static yuri::format_t get_format_from_string( std::string format, yuri::format_t group=YURI_FMT_NONE);
	EXPORT static yuri::format_t set_frame_from_mime(pBasicFrame frame, std::string mime);
protected:
	std::queue<pBasicFrame >  frames;
	Log log;
	mutex framesLock, notifyLock;
std::string name;
	yuri::format_t type;
	bool discrete, changed, notificationsEnabled, closed;
	yuri::size_t bytes, count, limit;
	yuri::size_t totalBytes, totalCount, dropped;
	yuri::ubyte_t dropPolicy;
	shared_array<int> notifySockets;
	weak_ptr<ThreadBase> source, target;
protected:
	virtual void do_set_changed(bool ch = true);
	virtual void do_clear_pipe();
	virtual void do_push_frame(pBasicFrame frame);
	virtual pBasicFrame do_pop_frame();
	virtual pBasicFrame do_pop_latest();
	virtual void do_set_limit(yuri::size_t l, yuri::ubyte_t policy);
	virtual yuri::format_t do_get_type();
	virtual void do_set_type(yuri::format_t type);
	virtual int do_get_notification_fd();
	virtual void do_cancel_notifications();

	virtual void do_notify();

protected:
/*	static std::map<std::string,yuri::format_t,compare_insensitive> mime_to_format;
	static std::map<yuri::format_tstd::string> format_to_mime;
	static void set_mime_types();
	static void add_to_formatsstd::string mime, yuri::format_t format);
	static mutex mime_conv_mutex;*/
};


}

}

#endif /* BASICPIPE_H_ */

