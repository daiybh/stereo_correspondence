/*!
 * @file 		RepackAudio.h
 * @author 		Zdenek Travnicek
 * @date		17.5.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef DUMMYMODULE_H_
#define DUMMYMODULE_H_

#include "yuri/core/IOThread.h"

namespace yuri {
namespace repack_audio {

class RepackAudio: public core::IOThread
{
public:
	IO_THREAD_GENERATOR_DECLARATION
	static core::pParameters configure();
	virtual ~RepackAudio();
private:
	RepackAudio(log::Log &log_, core::pwThreadBase parent, core::Parameters &parameters);
	virtual bool step();
	virtual bool set_param(const core::Parameter& param);
	size_t store_samples(const ubyte_t* start, size_t count);
	void push_current_frame();
	std::string dummy_name;
	std::vector<ubyte_t> samples_;
	size_t samples_missing_;
	size_t total_samples_;
	size_t channels_;
	format_t current_format_;
};

} /* namespace repack_audio */
} /* namespace yuri */
#endif /* DUMMYMODULE_H_ */
