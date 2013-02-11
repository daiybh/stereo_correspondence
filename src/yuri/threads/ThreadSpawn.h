#ifndef THREADSPAWN_H_
#define THREADSPAWN_H_
#include "yuri/threads/ThreadBase.h"
#include <boost/shared_ptr.hpp>

namespace yuri
{

namespace threads
{
using namespace std;
using boost::shared_ptr;

class ThreadBase;
class ThreadSpawn
{
public:
	ThreadSpawn(shared_ptr<ThreadBase> thread);
	virtual ~ThreadSpawn();
	void operator() ();
protected:
	shared_ptr<ThreadBase> thread;
};

}

}

#endif /*THREADSPAWN_H_*/
