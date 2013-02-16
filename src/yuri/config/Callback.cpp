/*!
 * @file 		Callback.cpp
 * @author 		Zdenek Travnicek
 * @date 		31.5.2008
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2008 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "Callback.h"

namespace yuri
{
namespace config
{

Callback::Callback(void (*func)(pThreadBase,pThreadBase),pThreadBase data):func(func),data(data)
{
}

Callback::~Callback()
{
}

void Callback::run(pThreadBase global)
{
	if (func) func(global,data);	
}

}
}
// End of File
