#ifndef EXCEPTION_H_
#define EXCEPTION_H_

#include <exception>
#include <string>
#include "yuri/io/types.h"
namespace yuri
{
namespace exception {

class Exception: public std::exception
{
public:
	EXPORT Exception();
	EXPORT Exception(std::string reason);
	EXPORT virtual ~Exception()  throw();
	EXPORT virtual const char* what() const throw();
protected:
	std::string reason;
};

}
}

#endif /*EXCEPTION_H_*/
