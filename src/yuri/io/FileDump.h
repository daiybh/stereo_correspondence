#ifndef FILEDUMP_H_
#define FILEDUMP_H_
#include "BasicIOThread.h"
#include <yuri/config/Config.h>
#include <fstream>
#include <string>
#include "yuri/config/Parameters.h"

namespace yuri
{
namespace io
{

using namespace yuri::config;

class FileDump: public BasicIOThread
{
public:
	FileDump(Log &log_,pThreadBase parent, Parameters &parameters);
	virtual ~FileDump();
	static shared_ptr<BasicIOThread> generate(Log &_log,pThreadBase parent,Parameters& parameters) throw (Exception);
	static shared_ptr<Parameters> configure();
protected:
	bool step();
protected:
	std::ofstream dump_file;
	std::string filename;

	int seq_chars;
	yuri::size_t seq_number, dumped_frames, dump_limit;


};

}
}
#endif /*FILEDUMP_H_*/

