#ifndef THREADCHILD_H_
#define THREADCHILD_H_
#include "boost/thread.hpp"
#include <yuri/io/types.h>
namespace yuri
{
namespace threads
{
class ThreadBase;
class EXPORT ThreadChild
{
public:
	ThreadChild(shared_ptr<boost::thread> thread, shared_ptr<ThreadBase> child, bool spawned=false);
	virtual ~ThreadChild();
	shared_ptr<boost::thread> thread_ptr;
	shared_ptr<ThreadBase> thread;
	bool finished;
	bool spawned;
};

}
}
#endif /*THREADCHILD_H_*/
