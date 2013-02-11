#include "Exception.h"

namespace yuri
{
namespace exception
{

Exception::Exception():reason("Generic exception")
{
}

Exception::Exception(std::string reason):reason(reason)
{

}
Exception::~Exception() throw()
{
}

const char *Exception::what() const throw()
{
	return (const char*) reason.c_str();
}

}
}

// End of File
