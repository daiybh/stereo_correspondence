#ifndef EXCEPTION_H_
#define EXCEPTION_H_

#include <exception>
#include <string>
#include "yuri/io/types.h"
namespace yuri
{
namespace exception {

class EXPORT Exception: public std::exception
{
public:
	Exception();
	Exception(std::string reason);
	virtual ~Exception()  throw();
	virtual const char* what() const throw();
protected:
	std::string reason;
};

}
}

#endif /*EXCEPTION_H_*/
