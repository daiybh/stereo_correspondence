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

#include "yuri/core/thread/ThreadChild.h"
#include "yuri/core/parameter/Parameters.h"
//#include "yuri/core/thread/ThreadSpawn.h"
#include <map>
#include <vector>
#include <atomic>
#include "yuri/core/utils/time_types.h"
#ifdef __linux__
#include <sys/time.h>
#else
#include <ctime>
typedef int pid_t;
#endif


//#define YURI_EXIT_OK				0
//#define YURI_EXIT_USER_BREAK		1
//#define YURI_EXIT_FINISHED			2

#define TRACE_METHOD log[log::trace] << __PRETTY_FUNCTION__ << " @ " << __FILE__ << ":" << __LINE__;

namespace yuri
{
namespace core
{

const int yuri_exit_finished = 0;
const int yuri_exit_interrupted = 1;

class EXPORT ThreadBase :public enable_shared_from_this<ThreadBase>
{
public:
	static Parameters			configure();
								ThreadBase(const log::Log &_log, pwThreadBase parent, const std::string& = std::string());
	virtual 					~ThreadBase() noexcept;
								ThreadBase(const ThreadBase&) = delete;
								ThreadBase(ThreadBase&&) = delete;
	void						operator=(const ThreadBase&) = delete;
	void						operator=(ThreadBase&&) = delete;

	void 						operator()();

	void 						child_ends(pwThreadBase child, int code);
	void 						finish() noexcept;
	void	 					request_end(int code = yuri_exit_finished);

//	virtual pid_t 				get_tid();
	/*!
	 * Returns true if the thread is currently running
	 * This basically means, that the 'operator()' method is executing
	 * @return true while the thread is running
	 */
	bool						running() const noexcept { return running_;}
private:
	virtual void 				run() = 0;
protected:
	//! Returns false if the thread should quit, true otherwise.
	//! The method also processes events and should be called occasionally
	bool 						still_running();
	//! Returns pointer to @em this.
	pThreadBase					get_this_ptr();
	//! Returns pointer to @em this.
	pcThreadBase				get_this_ptr() const;

	//! Adds and spawns new child thread
	bool 						spawn_thread(pThreadBase  thread);
	//! Adds new child thread without spawning it
	bool		 				add_child(pThreadBase  thread);


//	//! Return current thread id, if supported on current platform
//	virtual pid_t 				retrieve_tid();
	//! Prints own tid. Useful for debugging
	virtual void 				print_id(log::_debug_flags f=log::info);
	//! Sets CPU affinity to a single CPU core.
	virtual bool 				bind_to_cpu(size_t cpu);

/* ****************************************************************************
 *   Methods for managing thread hierarchy. Should not be called directly
 * ****************************************************************************/
protected:
	void		 				finish_thread(pwThreadBase child, bool join=false);
	void 						join_thread(pwThreadBase child);
	void 						finish_all_threads(bool join=false);
	void 						join_all_threads() noexcept;
	void 						request_finish_thread(pwThreadBase child);
	virtual void 				do_request_finish_thread(pwThreadBase child);
	void		 				process_pending_requests();
	virtual yuri::size_t 		get_active_childs();

	bool		 				set_params(const Parameters &parameters);
	virtual bool 				set_param(const Parameter &parameter);
	template<typename T> bool 	set_param(const std::string& name, const T& value);
	std::string					get_node_name() const;
private:
	bool 						do_spawn_thread(pThreadBase  thread);
	bool 						do_add_child(pThreadBase  thread, bool spawned=true);

	void 						do_finish_thread(pwThreadBase child, bool join=false) noexcept;
	void		 				do_join_thread(pwThreadBase child, bool check=true) noexcept;

//	virtual	bool				do_child_ended(size_t remaining_child_count);
	virtual void				child_ends_hook(pwThreadBase child, int code, size_t remaining_child_count);
protected:
	log::Log					log;
	pwThreadBase 				parent_;
	std::atomic<bool>			ending_;
//	bool					end_requested;
//	mutex 					end_lock;
	mutex						finish_lock;

	mutex 						children_lock_;
	std::map<pwThreadBase, ThreadChild, std::owner_less<pwThreadBase>>
								children_;

	mutex 						timer_lock;
	duration_t					join_timeout;
//	int 					exitCode;
//	bool 						finishWhenChildEnds;
//	bool					quitWhenChildsEnd;
//	pid_t 						own_tid;
private:
	std::vector<pwThreadBase>	ending_childs_;
	position_t	 				cpu_affinity_;
	std::atomic<bool>			running_;
	std::string 				node_id_;
	std::string					node_name_;

public:
	static void 				sleep (const duration_t& us);
};


template<typename T> bool ThreadBase::set_param(const std::string& name, const T& value)
{
	return set_param(Parameter (name,value));
}

}
}
#endif /*THREADBASE_H_*/
