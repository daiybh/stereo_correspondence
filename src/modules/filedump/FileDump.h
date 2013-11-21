/*!
 * @file 		FileDump.h
 * @author 		Zdenek Travnicek
 * @date 		24.7.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2010 - 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef FILEDUMP_H_
#define FILEDUMP_H_
#include "yuri/core/IOThread.h"
#include <fstream>
#include <string>

namespace yuri
{
namespace dump
{


class FileDump: public core::IOThread
{
public:
	FileDump(log::Log &log_, core::pwThreadBase parent, core::Parameters &parameters);
	virtual ~FileDump();
	static core::pIOThread generate(log::Log &_log,core::pwThreadBase parent, core::Parameters& parameters);
	static core::pParameters configure();
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

