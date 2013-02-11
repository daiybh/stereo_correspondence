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

class Callback
{
public:
	Callback(pCallback func,pThreadBase data);
	virtual ~Callback();
	virtual void run(pThreadBase global);
protected:
	pCallback func;
	pThreadBase data;
};

}
}
#endif /*CALLBACK_H_*/
