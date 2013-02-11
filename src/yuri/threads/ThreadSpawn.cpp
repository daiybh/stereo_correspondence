#include "ThreadSpawn.h"

namespace yuri
{

namespace threads
{

ThreadSpawn::ThreadSpawn(shared_ptr<ThreadBase> thread):thread(thread)
{
}

ThreadSpawn::~ThreadSpawn()
{
}

void ThreadSpawn::operator ()()
{
	if (!thread.get()) return;
	(*thread)();
	thread.reset();
}

}

}
