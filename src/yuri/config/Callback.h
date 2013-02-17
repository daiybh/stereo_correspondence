/*!
 * @file 		Callback.h
 * @author 		Zdenek Travnicek
 * @date 		31.5.2008
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2008 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef CALLBACK_H_
#define CALLBACK_H_

namespace yuri {
namespace config {
class Callback;
}}

#include "yuri/threads/ThreadBase.h"

namespace yuri
{
namespace config
{
using namespace yuri::threads;


typedef void (*pCallback)(pThreadBase,pThreadBase);

class EXPORT Callback
{
public:
								Callback(pCallback func,pThreadBase data);
	virtual 					~Callback();
	virtual void 				run(pThreadBase global);
protected:
	pCallback 					func;
	pThreadBase 				data;
};

}
}
#endif /*CALLBACK_H_*/
