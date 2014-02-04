/*!
 * @file 		FileDump.h
 * @author 		Zdenek Travnicek
 * @date 		24.7.2010
 * @date		04.2.2014
 * @copyright	Institute of Intermedia, CTU in Prague, 2010 - 2014
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef FILEDUMP_H_
#define FILEDUMP_H_
#include "yuri/core/thread/IOFilter.h"
#include <fstream>
#include <string>

namespace yuri
{
namespace dump
{


class FileDump: public core::IOFilter
{
public:
	FileDump(log::Log &log_,core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~FileDump() noexcept;
	//static core::pIOThread generate(log::Log &_log,core::pwThreadBase parent, core::Parameters& parameters);
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
private:
	virtual core::pFrame do_simple_single_step(const core::pFrame& frame) override;
	virtual bool set_param(const core::Parameter &param) override;
	std::ofstream dump_file;
	std::string filename;

	int seq_chars;
	size_t seq_number, dumped_frames, dump_limit;


};

}
}
#endif /*FILEDUMP_H_*/

