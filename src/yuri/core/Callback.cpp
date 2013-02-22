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
namespace core
{

Callback::Callback(pfCallback func,pwThreadBase data):func(func),data(data)
{
}

Callback::~Callback()
{
}

void Callback::run(pwThreadBase global)
{
	if (func) func(global,data);	
}

}
}
// End of File
