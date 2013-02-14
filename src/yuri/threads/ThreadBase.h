#ifndef THREADBASE_H_
#define THREADBASE_H_
#include <boost/weak_ptr.hpp>

namespace yuri
{
namespace threads
{
class ThreadBase;
using boost::weak_ptr;
typedef weak_ptr<class ThreadBase> pThreadBase;
}
}
#include "yuri/io/types.h"
#include "yuri/log/Log.h"
#include "yuri/threads/ThreadChild.h"
#include <boost/smart_ptr.hpp>
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

#include "yuri/threads/ThreadSpawn.h"

namespace yuri
{
namespace threads
{

using namespace yuri::threads;
using namespace yuri::log;




class ThreadBase :public boost::enable_shared_from_this<ThreadBase>
{
public:
	//ThreadBase(Log &_log, pThreadBase parent);
	ThreadBase(Log &_log, pThreadBase parent);
	virtual ~ThreadBase();
	virtual void operator()();
	//virtual void register_child(pThreadBase child,long id);
	virtual void child_ends(pThreadBase child, int code);
	virtual void finish();
	virtual void request_end();
	virtual pid_t get_tid();
protected:
	virtual void run()=0;
	//virtual void register_to_parent();
	//virtual boost::thread::thread *spawn_thread();
	virtual bool still_running();
	//virtual void init_thread() { }
	virtual bool spawn_thread(shared_ptr<ThreadBase>  thread);
	virtual bool add_child(shared_ptr<ThreadBase>  thread);
	virtual void finish_thread(pThreadBase child, bool join=false);
	virtual void join_thread(pThreadBase child);
	virtual void finish_all_threads(bool join=false);
	virtual void join_all_threads();
	virtual bool do_spawn_thread(shared_ptr<ThreadBase>  thread);
	virtual bool do_add_child(shared_ptr<ThreadBase>  thread, bool spawned=true);
	virtual void do_finish_thread(pThreadBase child, bool join=false);
	virtual void do_join_thread(pThreadBase child, bool check=true);
	/*virtual void start_timer();
	virtual void pause_timer();
	virtual void do_add_timer();*/
	virtual void request_finish_thread(pThreadBase child);
	virtual void do_request_finish_thread(pThreadBase child);
	virtual void do_pending_requests();
	virtual yuri::size_t get_active_childs();
	virtual weak_ptr<ThreadBase> get_this_ptr();
	virtual pid_t retrieve_tid();
	virtual void print_id(_debug_flags f=info);
	virtual bool bind_to_cpu(yuri::uint_t cpu);
	virtual void set_thread_name(std::string name);
protected:
	Log log;
	pThreadBase parent;
	bool end,end_requested;
	boost::mutex end_lock,finish_lock;
	boost::timed_mutex children_lock;
	std::map<pThreadBase, shared_ptr<ThreadChild> > children;
	boost::mutex timer_lock;
#ifdef __linux__
	struct timeval tv_start,tv_stop;
#endif
	yuri::size_t elsec,elusec;
	bool timer_started;
	boost::posix_time::time_duration join_timeout;
	int exitCode;
	threadId_t lastChild;
	bool finishWhenChildEnds, quitWhenChildsEnd;
	std::vector<pThreadBase> endingChilds;
	pid_t own_tid;
public: static long new_thread_id();
protected: static long last_user_thread_id;
protected: static boost::mutex last_user_thread_id_mutex;
public: static void sleep (unsigned long microseconds);
};


}
}
#endif /*THREADBASE_H_*/
