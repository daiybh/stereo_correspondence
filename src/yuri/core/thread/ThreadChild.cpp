/*!
 * @file 		ThreadChild.cpp
 * @author 		Zdenek Travnicek
 * @date 		24.7.2010
 * @date		21.11.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2010 - 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "ThreadChild.h"

namespace yuri
{
namespace core
{
	 
ThreadChild::ThreadChild():finished(true),spawned(false) {}

ThreadChild::ThreadChild(std::thread&& thread,
		pThreadBase child,bool spawned)
	:thread_ptr(std::move(thread)),thread(child),finished(false),spawned(spawned)
{
}

ThreadChild::ThreadChild(ThreadChild&& rhs)
	:thread_ptr(std::move(rhs.thread_ptr)),thread(std::move(rhs.thread)),finished(rhs.finished),spawned(rhs.spawned)
{
	rhs.finished 	= true;
	rhs.spawned  	= false;
}
ThreadChild& ThreadChild::operator=(ThreadChild&& rhs)
{
	thread_ptr		= std::move(rhs.thread_ptr);
	thread	 		= std::move(rhs.thread);
	finished		= rhs.finished;
	spawned			= rhs.spawned;
	rhs.finished 	= false;
	rhs.spawned 	= false;
	return *this;
}

ThreadChild::~ThreadChild() noexcept
{
}

}
}

// End of file
