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
#include "yuri/core/forward.h"
namespace yuri
{

namespace core
{


class EXPORT ThreadSpawn
{
public:
							ThreadSpawn(pThreadBase thread);
	virtual 				~ThreadSpawn();
	void 					operator() ();
protected:
	pThreadBase			 	thread_;
};

}

}

#endif /*THREADSPAWN_H_*/
