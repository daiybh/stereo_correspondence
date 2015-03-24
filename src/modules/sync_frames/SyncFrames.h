/*!
 * @file 		SyncFrames.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		23.03.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef SYNCFRAMES_H_
#define SYNCFRAMES_H_

#include "yuri/core/thread/MultiIOFilter.h"

namespace yuri {
namespace sync_frames {

class SyncFrames: public core::MultiIOFilter
{
	using base_type = core::MultiIOFilter;
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	SyncFrames(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~SyncFrames() noexcept;
private:
	virtual std::vector<core::pFrame> do_single_step(std::vector<core::pFrame> frames) override;
	virtual bool set_param(const core::Parameter& param) override;

	duration_t tolerance_;
};

} /* namespace sync_frames */
} /* namespace yuri */
#endif /* SYNCFRAMES_H_ */
