/*!
 * @file 		Convert.h
 * @author 		<Your name>
 * @date 		30.10.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#ifndef CONVERT_H_
#define CONVERT_H_

#include "yuri/core/thread/IOFilter.h"
#include "yuri/core/thread/ConverterThread.h"
namespace yuri {
namespace core {
class Convert;
typedef shared_ptr<Convert> pConvert;
class Convert: public core::IOFilter, public core::ConverterThread
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	Convert(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~Convert() noexcept;

	pFrame convert_to_any(const pFrame& frame, const std::vector<format_t>& fmts);

private:
	pFrame 	do_convert_frame(pFrame frame_in, format_t target_format);
	pFrame 	do_simple_single_step(const pFrame& frame);
	virtual bool set_param(const core::Parameter& param);
	format_t	format_;

	struct convert_pimpl_;
	unique_ptr<convert_pimpl_> pimpl_;
};

} /* namespace convert */
} /* namespace yuri */
#endif /* CONVERT_H_ */
