/*!
 * @file 		ThreadChild.h
 * @author 		Zdenek Travnicek
 * @date 		24.7.2010
 * @date		21.11.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2010 - 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef THREADCHILD_H_
#define THREADCHILD_H_
#include "yuri/core/forward.h"
namespace yuri
{
namespace core
{

class ThreadChild
{
public:
	EXPORT 						ThreadChild();
	EXPORT 						ThreadChild(yuri::thread&& thread,
			pThreadBase child, bool spawned=false);
	EXPORT						~ThreadChild() noexcept;
	EXPORT 						ThreadChild(const ThreadChild&)=delete;
	EXPORT 						ThreadChild(ThreadChild&& rhs);
	EXPORT ThreadChild&			operator=(ThreadChild&& rhs);
	yuri::thread 				thread_ptr;
	pThreadBase 				thread;
	bool 						finished;
	bool 						spawned;
};

}
}
#endif /*THREADCHILD_H_*/
