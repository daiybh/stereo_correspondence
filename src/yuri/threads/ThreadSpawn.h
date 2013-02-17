/*!
 * @file 		ThreadSpawn.h
 * @author 		Zdenek Travnicek
 * @date 		24.7.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */


#ifndef THREADSPAWN_H_
#define THREADSPAWN_H_
#include "yuri/threads/ThreadBase.h"
#include <boost/shared_ptr.hpp>

namespace yuri
{

namespace threads
{

class ThreadBase;
class EXPORT ThreadSpawn
{
public:
							ThreadSpawn(shared_ptr<ThreadBase> thread);
	virtual 				~ThreadSpawn();
	void 					operator() ();
protected:
	shared_ptr<ThreadBase> 	thread;
};

}

}

#endif /*THREADSPAWN_H_*/
