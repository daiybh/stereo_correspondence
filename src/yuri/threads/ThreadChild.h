#ifndef THREADCHILD_H_
#define THREADCHILD_H_
#include "boost/thread.hpp"
#include <boost/shared_ptr.hpp>
namespace yuri
{
namespace threads
{
using namespace std;
using boost::shared_ptr;
class ThreadBase;
class ThreadChild
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
