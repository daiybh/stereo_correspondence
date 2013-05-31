/*!
 * @file 		BasicPipe.h
 * @author 		Zdenek Travnicek
 * @date 		28.7.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef BASICPIPE_H_
#define BASICPIPE_H_
#include "yuri/core/platform.h"
#include <queue>
#include <string>
#include <map>
#include <cassert>
#include <sys/types.h>
#ifdef YURI_LINUX
#include <sys/socket.h>
#else
#ifdef YURI_WINDOWS
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
#include "yuri/core/Parameters.h"
#include "yuri/core/ThreadBase.h"

namespace yuri {

namespace core {

struct compare_insensitive {
	bool operator() (const std::string a, const std::string b) const
			{ return boost::ilexicographical_compare(a,b); }
};

class EXPORT BasicPipe {
public:
	BasicPipe					(log::Log &log_,std::string name);
	virtual 					~BasicPipe();

	void 						push_frame(pBasicFrame frame);

	pBasicFrame 				pop_frame();
	pBasicFrame 				pop_latest();

	void	 					set_type(yuri::format_t type);
	yuri::format_t				get_type();


	// TODO This should be also protected by a mutex
	yuri::size_t 				get_size() { return bytes; }
	yuri::size_t 				get_count() { return count; }

	virtual bool 				is_changed();
	virtual bool 				is_empty();

	virtual void 				close();
	virtual bool 				is_closed();

	virtual void 				set_limit(yuri::size_t limit0, yuri::ubyte_t policy=YURI_DROP_SIZE);

	virtual int 				get_notification_fd();
	virtual void 				cancel_notifications();

public:
	static shared_ptr<BasicPipe>
								generator(log::Log &log, std::string name, Parameters &parameters);
	static shared_ptr<Parameters>
								configure();
	static std::map<yuri::format_t, const FormatInfo_t>
								formats;
	static boost::mutex 		format_lock;
	static std::string 			get_type_string(yuri::format_t type);
	static std::string 			get_format_string(yuri::format_t type);
	static std::string 			get_simple_format_string(yuri::format_t type);
	static yuri::size_t 		get_bpp_from_format(yuri::format_t type);
	static FormatInfo_t 		get_format_info(yuri::format_t format);
	static yuri::format_t 		get_format_group(yuri::format_t format);
	static yuri::format_t 		get_format_from_string( std::string format, yuri::format_t group=YURI_FMT_NONE);
	static yuri::format_t 		set_frame_from_mime(pBasicFrame frame, std::string mime);
protected:
	std::queue<pBasicFrame >  	frames;
	log::Log 					log;
	mutex 						framesLock;
	mutex						notifyLock;
	std::string 				name;
	yuri::format_t 				type;
	bool 						discrete;
	bool						changed;
	bool						notificationsEnabled;
	bool						closed;
	yuri::size_t 				bytes;
	yuri::size_t				count;
	yuri::size_t				limit;
	yuri::size_t 				totalBytes;
	yuri::size_t				totalCount;
	yuri::size_t				dropped;
	yuri::ubyte_t 				dropPolicy;
	std::vector<int> 			notifySockets;
	pThreadBase			 		source;
	pThreadBase					target;
protected:
	virtual void 				do_set_changed(bool ch = true);
	virtual void 				do_clear_pipe();
	virtual void 				do_push_frame(pBasicFrame frame);
	virtual pBasicFrame 		do_pop_frame();
	virtual pBasicFrame 		do_pop_latest();
	virtual void 				do_set_limit(yuri::size_t l, yuri::ubyte_t policy);
	virtual yuri::format_t 		do_get_type();
	virtual void 				do_set_type(yuri::format_t type);
	virtual int 				do_get_notification_fd();
	virtual void 				do_cancel_notifications();

	virtual void 				do_notify();

};


}

}

#endif /* BASICPIPE_H_ */

