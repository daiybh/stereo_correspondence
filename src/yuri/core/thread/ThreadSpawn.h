/*!
 * @file 		ThreadSpawn.h
 * @author 		Zdenek Travnicek
 * @date 		24.7.2010
 * @date		21.11.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2010 - 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */


#ifndef THREADSPAWN_H_
#define THREADSPAWN_H_
#include "yuri/core/forward.h"

namespace yuri
{

namespace core
{

class ThreadSpawn
{
public:
	EXPORT					ThreadSpawn(pThreadBase thread);
	EXPORT 					~ThreadSpawn() noexcept;
	EXPORT void 			operator() ();
protected:
	pThreadBase			 	thread_;
};

}

}

#endif /*THREADSPAWN_H_*/
