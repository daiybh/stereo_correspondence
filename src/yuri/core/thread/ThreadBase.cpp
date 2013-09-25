/*!
 * @file 		ThreadBase.cpp
 * @author 		Zdenek Travnicek
 * @date 		24.7.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "ThreadBase.h"
#include "yuri/core/thread/ThreadSpawn.h"
#include <sys/types.h>
#include <string>
#ifdef YURI_LINUX
#include <pthread.h>
#include <sys/syscall.h>
#include <sys/prctl.h>
#include <unistd.h>
#endif
#include "yuri/core/utils/Timer.h"
#include "yuri/core/utils.h"
#include <cassert>
namespace yuri
{
namespace core
{

using namespace yuri::log;

Parameters ThreadBase::configure()
{
	Parameters p;
	p["cpu"]["Bind thread to cpu"]=-1;
	p["debug"]["Change debug level. value 0 will keep inherited value from app, lower numbers will reduce verbosity, higher numbers will make output more verbose."]=0;
	p["_node_name"]["Name of the node. Will be filled automatically by the builder."]="";
	return p;
}


ThreadBase::ThreadBase(const log::Log &_log, pwThreadBase parent, const std::string& id):log(_log),parent_(parent),
	ending_(false), join_timeout(2.5_s),
	/*lastChild(0),*//*finishWhenChildEnds(false),*//*quitWhenChildsEnd(true),*///own_tid(0),
	running_(false),node_id_(id)
{
//	log[debug] << "Parent " << (void *)(parent.lock().get()) << "\n";
}

ThreadBase::~ThreadBase() noexcept
{
	TRACE_METHOD
	join_all_threads();
}

void ThreadBase::operator()()
{
	TRACE_METHOD
	running_ = true;
	log[verbose_debug] << "Starting thread" << "\n";
	run();
	log[verbose_debug] << "Thread finished execution" << "\n";
	request_end();
	running_ = false;
	log[verbose_debug] << "Not running";
}


void ThreadBase::finish() noexcept
{
	TRACE_METHOD
	log[verbose_debug] << "finish()";
//	finish_all_threads();
	join_all_threads();
	ending_=true;
}

bool ThreadBase::still_running()
{
	TRACE_METHOD
	process_pending_requests();
	return !ending_;
}

void ThreadBase::child_ends(pwThreadBase child, int code)
{
	TRACE_METHOD
	log[verbose_debug] << "Received childs_end() from child with code "	<< code;
	{
//		lock_t _(children_lock_);
		child_ends_hook(child, code, children_.size());
	}
	request_finish_thread(child);
}

/// Method to request end of a thread
///
/// Method notifies it's parent, that it wants/needs to quit.
/// if there's no parent, it is finished immediately
void ThreadBase::request_end(int code)
{
	TRACE_METHOD
	log[verbose_debug] << "request_end(): " << code;
	ending_ = true;
	if (pThreadBase parent =  parent_.lock()) {
		log[log::debug] << "Notifying parent that about my end...";
		parent->child_ends(get_this_ptr(),code);
		parent_.reset(); // Parent knows I'm dead, so let's forget...
	}
//	finish();
}

/// Spawns new thread
///
/// @param thread instance of ThreadBase to be run in spawned thread
/// @return true on success, false otherwise
bool ThreadBase::spawn_thread(pThreadBase thread)
{
	TRACE_METHOD
	lock_t l(children_lock_);
	return do_spawn_thread(thread);
}

bool ThreadBase::add_child(pThreadBase thread)
{
	TRACE_METHOD
	lock_t l(children_lock_);
	return do_add_child(thread,false);
}
/// Finishes the thread
///
/// It calls finish() method of the child. Optionaly it can also join the thread.
/// @param child Thread to be finished
/// @param join should the thread be joind after finishing? default value is false
void ThreadBase::finish_thread(pwThreadBase child, bool join)
{
	TRACE_METHOD
	lock_t l(children_lock_);
	do_finish_thread(child,join);
}
/// Joins thread
///
/// Method blocks (obviously :)
/// If the thread to be joined is not finished, it will be finished automatically
/// @param thread Thread to be joined. (more precisely, pointer to object, which
///		is running in thread that is to be closed)
void ThreadBase::join_thread(pwThreadBase child)
{
	TRACE_METHOD
	lock_t l(children_lock_);
	do_join_thread(child,true);
}
/// Finish all
///
/// Finishes all threads. Optionally, it can also join the threads.
void ThreadBase::finish_all_threads(bool join)
{
	TRACE_METHOD
	lock_t l(children_lock_);
	log[debug] << "Finishing all childs" << "\n";
	for (auto& child: children_) {
		do_finish_thread(child.first, join);
	}
}
/// Joins all threads
///
/// Method blocks (obviously :)
/// \bug There may be problem with iteration over the std::map container while removing elements from it.
void ThreadBase::join_all_threads() noexcept
{
	TRACE_METHOD
	log[debug] << "Joining all childs";
	lock_t l(children_lock_);
	yuri::size_t s = children_.size();
	log[debug] << s << "\n";
	while (children_.size()) {
		log[verbose_debug] << "There's " << children_.size() << "threads left "
			<< "\n";
		auto child=children_.begin();
		do_join_thread(child->first,true);
	}
}

/// Actually spawns new thread
///
/// This method should not be called by itself, user should rather call spawn_thread in order to get proper locking
/// @param thread instance of ThreadBase to be run in spawned thread
/// @return true on success, false otherwise
bool ThreadBase::do_spawn_thread(pThreadBase thread)
{
	TRACE_METHOD
	return do_add_child(thread,true);
}
bool ThreadBase::do_add_child(pThreadBase thread, bool spawn)
{
	TRACE_METHOD
	assert(thread.get());
	yuri::thread t;
	if (spawn) {
		try {
			t = yuri::thread(ThreadSpawn(thread));
		}
		catch (std::runtime_error&) {
			log[log::info] << "Failed to spawn thread!!";
			return false;
		}
	}
	children_[thread] = std::move(ThreadChild(std::move(t), thread, spawn));
	return true;
}

void ThreadBase::do_finish_thread(pwThreadBase child, bool join) noexcept
{
	TRACE_METHOD
	log[debug] << "Finishing child";
	auto child_it = children_.find(child);
	if (child_it == children_.end()) {
		log[warning] << "Trying to finish unregistered child. ";
		return;
	}
	auto& child_ref = child_it->second;
	if (!child_ref.finished) {
		child_ref.thread->finish();
		child_ref.finished=true;
	}
	if (join) {
		if 	(child_ref.spawned) do_join_thread(child,false);
		else log[warning] << "Requested to join object that is not separate thread!";
	}
}

void ThreadBase::do_join_thread(pwThreadBase child, bool /*check*/) noexcept
{
	TRACE_METHOD
	log[debug] << "Joining child (" <<children_.size() << ")" << "\n";
//	if (check) {
	if (children_.find(child)==children_.end()) {
		log[warning] << "Trying to join unregistered child. " << "\n";
		//delete thread;
		return;
	}
	auto& child_ref = children_[child];
	if (!child_ref.finished) {
		do_finish_thread(child,false);
	}

	if (!child_ref.spawned) {
		log[warning] << "Requested to join object that is not separate thread!";
	} else {
		Timer timer;
		while (child_ref.thread->running() && timer.get_duration() < join_timeout) {
			sleep(100_ms);
			log[debug] << "Waiting for child: " << timer.get_duration();
		}
		if (!child_ref.thread->running()) {
			child_ref.thread_ptr.join();
		} else {
			log[warning] << "Failed to join thread " <<
					child_ref.thread_ptr.get_id() << " with " <<
						join_timeout << "timeout. Detaching thread";
			child_ref.thread_ptr.detach();
		}

//#endif
	}
	log[debug] << "Deleting child" << "\n";
	children_.erase(child);
//	if (do_child_ended(children_.size())) request_end(yuri_exit_finished);
	//delete thread;
	log[debug] << "Child joined (" << children_.size() << " left)" << "\n";
}

void ThreadBase::request_finish_thread(pwThreadBase child)
{
	TRACE_METHOD
//	lock_t l2(end_lock);
	lock_t l(children_lock_);

	do_request_finish_thread(child);
}
void ThreadBase::do_request_finish_thread(pwThreadBase child)
{
	TRACE_METHOD
	if (child.expired()) {
		log[log::error] << "Invalid child requested to be finished, this won't work";
	} else {
		log[log::verbose_debug] << "Adding a child to ending_childs";
		ending_childs_.push_back(child);
	}
}

void ThreadBase::process_pending_requests()
{
	TRACE_METHOD
//	lock_t _(children_lock);
	log[log::verbose_debug] << ending_childs_.size() << " childs wait for finishing";
	while (!ending_childs_.empty()) {
		pwThreadBase child = ending_childs_.back();
		ending_childs_.pop_back();

		if (!child.expired()) {
			log[debug] << "TB::ppr: Finishing child";

			finish_thread(child,true);
		} else {
			log[log::warning] << "Tried to finish an empty child";
		}
	}
}
// Return true if the app should quit
//bool ThreadBase::do_child_ended(size_t remaining_child_count)
//{
//	if (remaining_child_count == 0) return true;
//	return false;
//}
void ThreadBase::sleep (const duration_t& ms)
{
	std::this_thread::sleep_for(std::chrono::microseconds(ms));
}

yuri::size_t ThreadBase::get_active_childs()
{
	lock_t l(children_lock_);
	return children_.size();
}

pThreadBase ThreadBase::get_this_ptr()
{
	return shared_from_this();
}
pcThreadBase ThreadBase::get_this_ptr() const
{
	return shared_from_this();
}

void ThreadBase::child_ends_hook(pwThreadBase, int code, size_t remaining_child_count)
{
	TRACE_METHOD
	log[log::debug] << "Children hook, code " << code;
	// Default implementation handles only interrupted threads
	if (code == yuri_exit_interrupted) {
		log[log::debug] << "Requesting interrupt";
		request_end(code);
	} else if (code == yuri_exit_finished && remaining_child_count == 1) {
		// All childs ended
		log[log::debug] << "Requesting finish";
		request_end(yuri_exit_finished);
	}
}

//pid_t ThreadBase::get_tid()
//{
//	return own_tid;
//}
//
//pid_t ThreadBase::retrieve_tid()
//{
//#if defined(YURI_LINUX) && !defined(YURI_ANDROID)
//	own_tid = syscall(SYS_gettid);
//	return own_tid;
//#else
//	log[warning] << "TID not supported under this platform"<< "\n";
//	return 0;
//#endif
//}

void ThreadBase::print_id(_debug_flags /*f*/)
{
	//log[f] << "Started as thread " << retrieve_tid() <<"\n";
}

bool ThreadBase::bind_to_cpu(size_t cpu)
{
#if defined(YURI_LINUX) && !defined(YURI_ANDROID)
	cpu_set_t cpus;
	CPU_ZERO(&cpus);
	CPU_SET(cpu,&cpus);
	pthread_t thread = pthread_self();
	int ret = pthread_setaffinity_np(thread,sizeof(cpu_set_t),&cpus);
	if (ret) {
		log[error] << "Failed to set CPU affinity. Error " << ret << "\n";
		return false;
	}
	std::string cpu_s = "";
	for (int i = 0; i< CPU_SETSIZE; ++i) {
		if (CPU_ISSET(i,&cpus)) cpu_s += lexical_cast<std::string>(i)+std::string(" ");
	}
	log[info] << "CPU affinity set for CPUS: " << cpu_s << "\n";
#else
	log[warning] << "Setting CPU affinity not supported under this platform" << "\n";
#endif
	return true;
}

bool ThreadBase::set_params(const Parameters &parameters)
{
	bool all_ok = true;
	for (const auto& param_pair: parameters) {
		const auto& parameter = param_pair.second;
		try {
			if (!set_param(parameter)) {
				log[log::warning] << "Error setting parameter " << parameter.get_name();
				all_ok = false;
			}
		}
		catch (bad_lexical_cast& e) {
			log[log::warning] << "Error processing parameter " << parameter.get_name() << ": " << e.what();
			all_ok = false;
		}
	}
	return all_ok;
}
bool ThreadBase::set_param(const Parameter &parameter)
{
	if (parameter.get_name() == "cpu") {
		cpu_affinity_=parameter.get<yuri::ssize_t>();
	} else if (parameter.get_name() == "debug") {
		/*int debug;
		debug = parameter.get<yuri::sint_t>();
		if (debug) {
			yuri::sint_t orig = log.get_flags();
			log.set_flags(((orig&flag_mask)<<debug)|(orig&~flag_mask));
		}*/
	} else if (parameter.get_name() == "_node_name") {
		node_name_ = parameter.get<std::string>();
		log.set_label(get_node_name());
//		set_log_id();
	} else return false;
	return true;
}
std::string ThreadBase::get_node_name() const
{
	if (node_name_.empty()) {
		return std::string("[")+node_id_+"] ";
	} else {
		return std::string("[")+node_id_+"/"+node_name_+"] ";
	}
}
}
}

// End of File


