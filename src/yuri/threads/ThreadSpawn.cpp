/*!
 * @file 		ThreadSpawn.cpp
 * @author 		Zdenek Travnicek
 * @date 		24.7.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

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
