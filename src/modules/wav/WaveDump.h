/*!
 * @file 		WaveDump.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		20.03.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef WAVEDUMP_H_
#define WAVEDUMP_H_

#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/RawAudioFrame.h"
#include "WaveHeader.h"

namespace yuri {
namespace wav {

class WaveDump: public core::SpecializedIOFilter<core::RawAudioFrame>
{
	using base_type = core::SpecializedIOFilter<core::RawAudioFrame>;
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	WaveDump(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~WaveDump() noexcept;
private:
	virtual core::pFrame do_special_single_step(core::pRawAudioFrame frame) override;
	virtual bool set_param(const core::Parameter& param) override;

	std::string filename_;
	std::ofstream file_;
	wav_header_t header_;

	bool format_set_;

};

} /* namespace wav */
} /* namespace yuri */
#endif /* WAVEDUMP_H_ */
