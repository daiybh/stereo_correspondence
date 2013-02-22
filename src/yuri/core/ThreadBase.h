/*!
 * @file 		ThreadBase.h
 * @author 		Zdenek Travnicek
 * @date 		24.7.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef THREADBASE_H_
#define THREADBASE_H_
#include "yuri/core/forward.h"
#include "yuri/log/Log.h"

#include "yuri/core/ThreadChild.h"
#include "yuri/core/ThreadSpawn.h"
#include <boost/enable_shared_from_this.hpp>
#include <map>
#include <vector>
#ifdef __linux__
#include <sys/time.h>
#else
#include <ctime>
typedef int pid_t;
#endif

// Types of threads spawned from yuri application.
// For user defined thread types use ThreadBase::new_thread_id();

#define YURI_THREAD_GENERIC 		0
#define YURI_THREAD_DRAW 			1
#define YURI_THREAD_RENDERER		2

#define YURI_THREAD_FILESOURCE		10
#define YURI_THREAD_FILESINK		11
#define YURI_THREAD_ENCODER			12

#define YURI_THREAD_USER 			0xffff

#define YURI_CALLBACK_GENERIC 		0
#define YURI_CALLBACK_DRAW			1
#define YURI_CALLBACK_DRAW_LEFT		1
#define YURI_CALLBACK_DRAW_RIGHT	2


#define YURI_CALLBACK_USER			0xffff


#define YURI_EXIT_OK				0
#define YURI_EXIT_USER_BREAK		1
#define YURI_EXIT_FINISHED			2

namespace yuri
{
namespace core
{


class EXPORT ThreadBase :public boost::enable_shared_from_this<ThreadBase>
{
public:
							ThreadBase(log::Log &_log, pwThreadBase parent);
	virtual 				~ThreadBase();

	virtual void 			operator()();
	virtual void 			child_ends(pwThreadBase child, int code);
	virtual void 			finish();
	virtual void 			request_end();
	virtual pid_t 			get_tid();
protected:
	virtual void 			run()=0;
	//! Returns false is the thread should quit, true otherwise.
	//! The method also processes events and should be called frequently
	virtual bool 			still_running();
	//! Adds and spawns new child thread
	virtual bool 			spawn_thread(pThreadBase  thread);
	//! Adds new child thread without spawning it
	virtual bool 			add_child(pThreadBase  thread);
	//! Returns pointer to @em this.
	virtual pwThreadBase	get_this_ptr();
	//! Return current thread id, if supported on current platform
	virtual pid_t 			retrieve_tid();
	//! Prints own tid. Useful for debugging
	virtual void 			print_id(log::_debug_flags f=log::info);
	//! Sets CPU affinity to a single CPU core.
	virtual bool 			bind_to_cpu(yuri::uint_t cpu);

/* ****************************************************************************
 *   Methods for managing thread hierarchy. Should not be called directly
 * ****************************************************************************/

	virtual void		 	finish_thread(pwThreadBase child, bool join=false);
	virtual void 			join_thread(pwThreadBase child);
	virtual void 			finish_all_threads(bool join=false);
	virtual void 			join_all_threads();
	virtual bool 			do_spawn_thread(pThreadBase  thread);
	virtual bool 			do_add_child(pThreadBase  thread, bool spawned=true);
	virtual void 			do_finish_thread(pwThreadBase child, bool join=false);
	virtual void 			do_join_thread(pwThreadBase child, bool check=true);
	virtual void 			request_finish_thread(pwThreadBase child);
	virtual void 			do_request_finish_thread(pwThreadBase child);
	virtual void 			do_pending_requests();
	virtual yuri::size_t 	get_active_childs();

	virtual void 			set_thread_name(std::string name);
protected:
	log::Log				log;
	pwThreadBase 			parent;
	bool 					end;
	bool					end_requested;
	boost::mutex 			end_lock;
	boost::mutex			finish_lock;
	boost::timed_mutex 		children_lock;
	std::map<pwThreadBase, pThreadChild >
							children;
	boost::mutex 			timer_lock;
#ifdef __linux__
	struct timeval 			tv_start;
	struct timeval			tv_stop;
#endif
	yuri::size_t 			elsec;
	yuri::size_t			elusec;
	bool 					timer_started;
	boost::posix_time::time_duration
							join_timeout;
	int 					exitCode;
	threadId_t 				lastChild;
	bool 					finishWhenChildEnds;
	bool					quitWhenChildsEnd;
	std::vector<pwThreadBase>
							endingChilds;
	pid_t 					own_tid;
public:
	static long 			new_thread_id();
protected:
	static long 			last_user_thread_id;
protected:
	static yuri::mutex 	last_user_thread_id_mutex;
public:
	static void 			sleep (unsigned long microseconds);
};


}
}
#endif /*THREADBASE_H_*/
