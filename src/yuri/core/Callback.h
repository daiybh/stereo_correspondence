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
#include "yuri/core/forward.h"

namespace yuri
{
namespace core
{

typedef yuri::shared_ptr<class Callback> pCallback;
typedef void (*pfCallback)(pwThreadBase,pwThreadBase);

class EXPORT Callback
{
public:
								Callback(pfCallback func,pwThreadBase data);
	virtual 					~Callback();
	virtual void 				run(pwThreadBase global);
protected:
	pfCallback 					func;
	pwThreadBase 				data;
};

}
}
#endif /*CALLBACK_H_*/
