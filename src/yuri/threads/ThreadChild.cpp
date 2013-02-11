#include "ThreadChild.h"

namespace yuri
{
namespace threads
{
	 
ThreadChild::ThreadChild(shared_ptr<boost::thread> thread,
		shared_ptr<ThreadBase> child,bool spawned)
	:thread_ptr(thread),thread(child),finished(false),spawned(spawned)
{
}

ThreadChild::~ThreadChild()
{
}

}
}

// End of file
