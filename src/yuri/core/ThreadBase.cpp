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
#include <sys/types.h>
#include <string>
#ifdef YURI_LINUX
#include <pthread.h>
#include <sys/syscall.h>
#include <sys/prctl.h>
#include <unistd.h>
#endif
namespace yuri
{
namespace core
{

using namespace yuri::log;

long ThreadBase::last_user_thread_id=YURI_THREAD_USER;
yuri::mutex ThreadBase::last_user_thread_id_mutex;

long ThreadBase::new_thread_id()
{
	lock l(last_user_thread_id_mutex);
	return ++last_user_thread_id;
}
ThreadBase::ThreadBase(log::Log &_log, pwThreadBase parent):log(_log),parent(parent),
	end(false),end_requested(false),
	elsec(0),elusec(0),timer_started(false),
#ifndef YURI_USE_CXX11
	join_timeout(boost::posix_time::milliseconds(2500)),
#else
	join_timeout(2500000),
#endif
	exitCode(YURI_EXIT_OK),
	lastChild(0),finishWhenChildEnds(false),quitWhenChildsEnd(true),own_tid(0)
{
	log[debug] << "Parent " << (void *)(parent.lock().get()) << "\n";
}

ThreadBase::~ThreadBase()
{
	log[verbose_debug] << "ThreadBase::~ThreadBase run " <<
		(double)((double)elsec * 1000.0 + ((double)elusec / 1000.0))
		<< "ms " << "\n";
	join_all_threads();
}

void ThreadBase::operator()()
{
	log[verbose_debug] << "Starting thread" << "\n";
	run();
	if (!end_requested) request_end();
}


void ThreadBase::finish()
{
	lock l(end_lock);
	// Death of a child is fatal
	log[verbose_debug] << "finish()" << "\n";
	log[debug] << "Finishind all threads" << "\n";
	finish_all_threads();
	end=true;
}

bool ThreadBase::still_running()
{
	lock l(end_lock);
	do_pending_requests();
	return !end && !end_requested;
}

void ThreadBase::child_ends(pwThreadBase child, int code)
{
	log[verbose_debug] << "Received childs_end() from child with code "
			<< code << "\n";
	// Death of a child is by default fatal
	if (finishWhenChildEnds) request_end();
	else switch (code) {
		case YURI_EXIT_OK: request_finish_thread(child); break;
		case YURI_EXIT_USER_BREAK:
		default:request_end();
	}
}

/// Method to request end of a thread
///
/// Method notifie it's parent, that it wants/needs to quit.
/// if there's no parent, it is finished immediately
void ThreadBase::request_end(int code)
{
	log[verbose_debug] << "request_end(): " << code;
	exitCode = code;
	lock l(end_lock);
	if (end || end_requested) return;
	end_requested=true;
	log[verbose_debug] << "end_request placed" << "\n";
	l.unlock();
	log[debug] << "I " << (parent.lock()?"":"don't ") << "have parent!" << "\n";
	if (!parent.expired()) {
		pThreadBase par =  parent.lock();
		if (par) par->child_ends(get_this_ptr(),exitCode);
		else finish();
	}
	else {
		finish();
	}
}

/// Spawns new thread
///
/// @param thread instance of ThreadBase to be run in spawned thread
/// @return true on success, false otherwise
bool ThreadBase::spawn_thread(pThreadBase thread)
{
	timed_lock l(children_lock);
	return do_spawn_thread(thread);
}

bool ThreadBase::add_child(pThreadBase thread)
{
	timed_lock l(children_lock);
	return do_add_child(thread,false);
}
/// Finished the thread
///
/// It calls finish() method of the child. Optionaly it can also join the thread.
/// @param thread Thread to be finished
/// @param join should the thread be joind after finishing? default value is false
void ThreadBase::finish_thread(pwThreadBase child, bool join)
{
	timed_lock l(children_lock);
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
	timed_lock l(children_lock);
	do_join_thread(child,true);
}
/// Finish all
///
/// Finishes all threads. Optionaly, it cas also join the threads.
void ThreadBase::finish_all_threads(bool join)
{
	timed_lock l(children_lock);
	log[debug] << "Finishing all childs" << "\n";
	for (std::map<pwThreadBase,pThreadChild >::iterator i = children.begin();
		i != children.end(); ++i) {
		do_finish_thread((*i).first,join);
	}
}
/// Joins all threads
///
/// Method blocks (obviously :)
/// \bug There may be problem with iteration over the std::map container while removing elements from it.
void ThreadBase::join_all_threads()
{
	log[debug] << "Joining all childs" << "\n";
	timed_lock l(children_lock);
	std::map<pwThreadBase,pThreadChild >::iterator i;
	yuri::size_t s = children.size();
	log[debug] << s << "\n";
	while (children.size()) {
		log[verbose_debug] << "There's " << children.size() << "threads left "
			<< "\n";
		i=children.begin();
		do_join_thread((*i).first,true);
	}
}

/// Actually spawns new thread
///
/// This method should not be called by itself, user should rather call spawn_thread in order to get proper locking
/// @param thread instance of ThreadBase to be run in spawned thread
/// @return true on success, false otherwise
bool ThreadBase::do_spawn_thread(pThreadBase thread)
{
	return do_add_child(thread,true);
}
bool ThreadBase::do_add_child(pThreadBase thread, bool spawned)
{
	assert(thread.get());
	pThreadChild child;
	shared_ptr<yuri::thread> t;
	if (spawned) {

		try {
			t.reset(new yuri::thread(ThreadSpawn(thread)));
		}
		catch (std::runtime_error&) {
			return false;
		}
		if (!t.get()) return false;
		child.reset(new ThreadChild(t,thread,true));
	}
	else child.reset(new ThreadChild(t,thread,false));
	children[pThreadBase(thread)]=child;
	//childMap[thread.get()] = lastChild++;
	return true;
}

void ThreadBase::do_finish_thread(pwThreadBase child, bool join)
{
	log[debug] << "Finishing child" << "\n";
	if (children.find(child)==children.end()) {
		log[warning] << "Trying to finish unregistered child. " << "\n";
		return;
	}
	if (!children[child]->finished) {
		children[child]->thread->finish();
		children[child]->finished=true;
//		if (children[child]->thread_ptr)
//			children[child]->thread_ptr->interrupt();
	}
	if (join) {
		if 	(children[child]->spawned) do_join_thread(child,false);
		else log[warning] <<
			"Requested to join object that is not separate thread!" <<
			"\n";
	}
}

void ThreadBase::do_join_thread(pwThreadBase child, bool check)
{
	log[debug] << "Joining child (" <<children.size() << ")" << "\n";
	if (check) {
		if (children.find(child)==children.end()) {
			log[warning] << "Trying to join unregistered child. " << "\n";
			//delete thread;
			return;
		}
		if (!children[child]->finished) {
			do_finish_thread(child,false);
		}

	}
	if (!children[child]->spawned) {
		log[warning] << "Requested to join object that is not separate thread!"
			<< "\n";
	} else {
#ifndef YURI_USE_CXX11
		if (!children[child]->thread_ptr->timed_join(join_timeout)) {
			log[warning] << "Failed to join thread " <<
				children[child]->thread_ptr->get_id() << " with " <<
			join_timeout.total_milliseconds() << "ms timeout." <<

				"Detaching thread." << "\n";
			children[child]->thread_ptr->detach();
		}
#else
		children[child]->thread_ptr->join();
#endif
	}
	log[debug] << "Deleting child" << "\n";
	children.erase(child);
	//delete thread;
	log[debug] << "Child joined (" << children.size() << " left)" << "\n";
}

void ThreadBase::request_finish_thread(pwThreadBase child)
{
	lock l2(end_lock);
	timed_lock l(children_lock);


	do_request_finish_thread(child);
}
void ThreadBase::do_request_finish_thread(pwThreadBase child)
{
	assert(!child.expired());
	/*if (childMap.find(child)==childMap.end()) {
		log[warning] << "Requested to finish non-registered child!" << "\n";
		return;
	}*/
	endingChilds.push_back(child);
}

void ThreadBase::do_pending_requests()
{
	while (!endingChilds.empty()) {
		pwThreadBase child = endingChilds.back();
		endingChilds.pop_back();
		log[debug] << "Killing child " << "\n";
		finish_thread(child,true);
	}
}
void ThreadBase::sleep (unsigned long ms)
{
	this_thread::sleep(microseconds(ms));
}

yuri::size_t ThreadBase::get_active_childs()
{
	timed_lock l(children_lock);
	return children.size();
}

pwThreadBase ThreadBase::get_this_ptr()
{
	pwThreadBase ptr (shared_from_this());
	return ptr;
}
pid_t ThreadBase::get_tid()
{
	return own_tid;
}

pid_t ThreadBase::retrieve_tid()
{
#if defined(YURI_LINUX) && !defined(YURI_ANDROID)
	own_tid = syscall(SYS_gettid);
	return own_tid;
#else
	log[warning] << "TID not supported under this platform"<< "\n";
	return 0;
#endif
}

void ThreadBase::print_id(_debug_flags f)
{
	log[f] << "Started as thread " << retrieve_tid() <<"\n";
}

bool ThreadBase::bind_to_cpu(yuri::uint_t cpu)
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

void ThreadBase::set_thread_name(std::string name)
{
#ifdef YURI_LINUX
	int ret = prctl(PR_SET_NAME,name.c_str(),0,0,0);
	log[info] << "Name set to " << name << " (result " << ret << ")"<<"\n";
#endif
}

}
}

// End of File


