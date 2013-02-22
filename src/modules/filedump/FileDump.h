/*!
 * @file 		FileDump.h
 * @author 		Zdenek Travnicek
 * @date 		24.7.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef FILEDUMP_H_
#define FILEDUMP_H_
#include "yuri/core/BasicIOThread.h"
#include <fstream>
#include <string>

namespace yuri
{
namespace dump
{


class FileDump: public core::BasicIOThread
{
public:
	FileDump(log::Log &log_, core::pwThreadBase parent, core::Parameters &parameters);
	virtual ~FileDump();
	static core::pBasicIOThread generate(log::Log &_log,core::pwThreadBase parent, core::Parameters& parameters);
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

