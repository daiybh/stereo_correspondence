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
#include "yuri/core/thread/ThreadBase.h"
namespace yuri
{

namespace core
{

ThreadSpawn::ThreadSpawn(pThreadBase thread):thread_(thread)
{
}

ThreadSpawn::~ThreadSpawn() noexcept
{
}

void ThreadSpawn::operator ()()
{
	if (!thread_) return;
	(*thread_)();
	thread_.reset(); // Delete the pointer so the object can't be called again
}

}

}
