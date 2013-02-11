#include "WindowBase.h"

namespace yuri
{
namespace graphics
{
using namespace yuri::log;
using yuri::threads::ThreadBase;
using yuri::threads::pThreadBase;

WindowBase::WindowBase(Log &log_, pThreadBase parent, Parameters &p):
	ThreadBase(log_,parent),x(0),y(0),width(0),height(0),name("noname")
{
	params.merge(p);
}

WindowBase::~WindowBase()
{
}













}
}

// End of File
