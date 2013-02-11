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
