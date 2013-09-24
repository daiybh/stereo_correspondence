/*!
 * @file 		ThreadChild.h
 * @author 		Zdenek Travnicek
 * @date 		24.7.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef THREADCHILD_H_
#define THREADCHILD_H_
#include "yuri/core/forward.h"
namespace yuri
{
namespace core
{

class EXPORT ThreadChild
{
public:
								ThreadChild();
								ThreadChild(yuri::thread&& thread,
			pThreadBase child, bool spawned=false);
								~ThreadChild() noexcept;
								ThreadChild(const ThreadChild&)=delete;
								ThreadChild(ThreadChild&& rhs);
	ThreadChild&				operator=(ThreadChild&& rhs);
	yuri::thread 				thread_ptr;
	pThreadBase 				thread;
	bool 						finished;
	bool 						spawned;
};

}
}
#endif /*THREADCHILD_H_*/
