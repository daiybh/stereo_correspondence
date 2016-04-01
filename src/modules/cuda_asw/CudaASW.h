/*!
 * @file 		CudaASW.h
 * @author 		Your name <lhotamir@fit.cvut.cz>
 * @date 		10.03.2016
 * @copyright	Institute of Intermedia, CTU in Prague, 2016
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef CUDAASW_H_
#define CUDAASW_H_

#include "yuri/core/thread/SpecializedMultiIOFilter.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "yuri/core/thread/Convert.h"

namespace yuri {
namespace cuda_asw {

class CudaASW: public core::SpecializedMultiIOFilter<core::RawVideoFrame, core::RawVideoFrame>
{
    using base_type = core::SpecializedMultiIOFilter<core::RawVideoFrame, core::RawVideoFrame>;
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	CudaASW(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~CudaASW() noexcept;
private:
	
	virtual std::vector<core::pFrame> do_special_step(std::tuple<core::pRawVideoFrame, core::pRawVideoFrame> frames) override;
	virtual bool set_param(const core::Parameter& param) override;
        int num_disparities;
        int iterations;
        int fill_iterations;
        std::shared_ptr<core::Convert> converter_left;
        std::shared_ptr<core::Convert> converter_right;
        std::vector<format_t>	supported_formats_;
};

} /* namespace cuda_asw */
} /* namespace yuri */
#endif /* CUDAASW_H_ */
